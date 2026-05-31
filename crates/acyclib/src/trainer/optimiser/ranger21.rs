use std::{
    collections::HashMap,
    fs::File,
    io::{BufRead, BufReader, Write},
    sync::Arc,
};

use crate::device::{
    operation::{AdamConfig, BaseOperations},
    tensor::DenseMatrix,
    Device, OperationError,
};

use super::{utils, OptimiserState};

#[derive(Clone, Copy, Debug)]
pub struct Ranger21Params {
    pub decay: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub alpha: f32,
    pub lookahead_mergetime: usize,
    pub clip: Option<(f32, f32)>,
}

impl Default for Ranger21Params {
    fn default() -> Self {
        Self { decay: 0.0, beta1: 0.9, beta2: 0.999, eps: 1.0e-7, alpha: 0.5, lookahead_mergetime: 5, clip: None }
    }
}

pub struct Ranger21<D: Device> {
    momentum: DenseMatrix<D>,
    neg_momentum: DenseMatrix<D>,
    velocity: DenseMatrix<D>,
    slow_params: DenseMatrix<D>,
    params: Ranger21Params,
    step: usize,
    lookahead_step: usize,
    slow_initialised: bool,
}

impl<D: Device> OptimiserState<D> for Ranger21<D> {
    type Params = Ranger21Params;

    fn new(device: Arc<D>, size: usize, params: Self::Params) -> Result<Self, D::DeviceError> {
        Ok(Self {
            momentum: DenseMatrix::zeroed(device.clone(), size, None)?,
            neg_momentum: DenseMatrix::zeroed(device.clone(), size, None)?,
            velocity: DenseMatrix::zeroed(device.clone(), size, None)?,
            slow_params: DenseMatrix::zeroed(device, size, None)?,
            params,
            step: 0,
            lookahead_step: 0,
            slow_initialised: false,
        })
    }

    fn update(
        &mut self,
        weights: &mut DenseMatrix<D>,
        grads: &mut DenseMatrix<D>,
        gradient_factor: f32,
        learning_rate: f32,
    ) -> Result<(), OperationError<D::DeviceError>> {
        assert!(weights.batch_size().is_none());
        assert!(self.momentum.batch_size().is_none());
        assert!(self.neg_momentum.batch_size().is_none());
        assert!(self.velocity.batch_size().is_none());
        assert_eq!(weights.size(), self.momentum.size());
        assert_eq!(weights.size(), self.neg_momentum.size());
        assert_eq!(weights.size(), self.velocity.size());

        if !self.slow_initialised {
            self.slow_params.copy_from(weights)?;
            self.slow_initialised = true;
        }

        self.step += 1;

        let params = self.params;
        let step = self.step as f32;
        let bias_correction1 = 1.0 - params.beta1.powf(step);
        let bias_correction2 = 1.0 - params.beta2.powf(step);
        let noise_norm = ((1.0 + params.beta2).powi(2) + params.beta2.powi(2)).sqrt();
        let learning_rate = learning_rate * bias_correction2.sqrt() / (bias_correction1 * noise_norm);
        let eps = params.eps * bias_correction2.sqrt();

        let cfg = AdamConfig {
            beta1: params.beta1 * params.beta1,
            beta2: params.beta2,
            gradient_factor,
            learning_rate,
            eps,
            denom: true,
            decay: 1.0 - params.decay * learning_rate,
            clip: params.clip,
        };

        let momentum = if self.step % 2 == 1 { &mut self.momentum } else { &mut self.neg_momentum };
        weights.buf.adam(&cfg, weights.size(), &grads.buf, &mut momentum.buf, &mut self.velocity.buf)?;

        self.lookahead_step += 1;
        if self.lookahead_step >= params.lookahead_mergetime {
            self.lookahead_step = 0;
            weights.lerp(1.0 - params.alpha, &self.slow_params)?;
            self.slow_params.copy_from(weights)?;
        }

        Ok(())
    }

    fn reset(&mut self) -> Result<(), D::DeviceError> {
        self.step = 0;
        self.lookahead_step = 0;
        self.slow_initialised = false;
        self.momentum.set_to(0.0)?;
        self.neg_momentum.set_to(0.0)?;
        self.velocity.set_to(0.0)?;
        self.slow_params.set_to(0.0)
    }

    fn set_params(&mut self, params: Self::Params) {
        self.params = params;
    }

    fn load_from_checkpoint(
        map: &mut HashMap<String, &mut Self>,
        path: &str,
        old_format: bool,
    ) -> Result<(), OperationError<D::DeviceError>> {
        let paths = [
            format!("{path}/momentum.bin"),
            format!("{path}/neg_momentum.bin"),
            format!("{path}/velocity.bin"),
            format!("{path}/slow.bin"),
        ];
        let mut momentum = utils::load_weights_from_file(&paths[0], old_format);
        let mut neg_momentum = utils::load_weights_from_file(&paths[1], old_format);
        let mut velocity = utils::load_weights_from_file(&paths[2], old_format);
        let mut slow = utils::load_weights_from_file(&paths[3], old_format);

        momentum.sort_by_key(|(id, _)| id.clone());
        neg_momentum.sort_by_key(|(id, _)| id.clone());
        velocity.sort_by_key(|(id, _)| id.clone());
        slow.sort_by_key(|(id, _)| id.clone());

        for ((((id1, mom), (id2, neg_mom)), (id3, vel)), (id4, slow_params)) in
            momentum.iter().zip(neg_momentum.iter()).zip(velocity.iter()).zip(slow.iter())
        {
            assert_eq!(id1, id2);
            assert_eq!(id1, id3);
            assert_eq!(id1, id4);

            let single = map.get_mut(id1).unwrap();
            single.momentum.load_from_slice(None, mom)?;
            single.neg_momentum.load_from_slice(None, neg_mom)?;
            single.velocity.load_from_slice(None, vel)?;
            single.slow_params.load_from_slice(None, slow_params)?;
            single.slow_initialised = true;
        }

        let step_path = format!("{path}/step_ranger21.txt");
        if let Ok(file) = File::open(&step_path) {
            for line in BufReader::new(file).lines() {
                let line = line.unwrap();
                let mut split = line.split(',');
                let id = split.next().unwrap().to_string();
                let step: usize = split.next().unwrap().parse().unwrap();
                let lookahead_step: usize = split.next().unwrap().parse().unwrap();
                if let Some(single) = map.get_mut(&id) {
                    single.step = step;
                    single.lookahead_step = lookahead_step;
                }
            }
        } else {
            let mut map = map.iter_mut().map(|(id, single)| (id.clone(), &mut single.step)).collect();
            load_legacy_step_file(&mut map, path);
        }

        Ok(())
    }

    fn write_to_checkpoint(map: &HashMap<String, &Self>, path: &str) -> Result<(), D::DeviceError> {
        let momentum: Vec<_> = map.iter().map(|(id, single)| (id, &single.momentum)).collect();
        let neg_momentum: Vec<_> = map.iter().map(|(id, single)| (id, &single.neg_momentum)).collect();
        let velocity: Vec<_> = map.iter().map(|(id, single)| (id, &single.velocity)).collect();
        let slow: Vec<_> = map.iter().map(|(id, single)| (id, &single.slow_params)).collect();
        utils::write_weights_to_file(&momentum, &format!("{path}/momentum.bin"))?;
        utils::write_weights_to_file(&neg_momentum, &format!("{path}/neg_momentum.bin"))?;
        utils::write_weights_to_file(&velocity, &format!("{path}/velocity.bin"))?;
        utils::write_weights_to_file(&slow, &format!("{path}/slow.bin"))?;

        let mut file = File::create(format!("{path}/step_ranger21.txt")).unwrap();
        for (id, single) in map {
            writeln!(file, "{id},{},{}", single.step, single.lookahead_step).unwrap();
        }

        Ok(())
    }
}

fn load_legacy_step_file(map: &mut HashMap<String, &mut usize>, path: &str) {
    if let Ok(file) = File::open(format!("{path}/step.txt")) {
        for line in BufReader::new(file).lines() {
            let line = line.unwrap();
            let mut split = line.split(',');
            let id = split.next().unwrap().to_string();
            let step: usize = split.next().unwrap().parse().unwrap();
            if let Some(single) = map.get_mut(&id) {
                **single = step;
            }
        }
    }
}
