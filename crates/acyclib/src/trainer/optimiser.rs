pub mod adam;
pub mod clip;
pub mod decay;
pub mod radam;
pub mod ranger;
pub mod ranger21;
pub mod utils;

use std::{collections::HashMap, fmt::Debug, marker::PhantomData, sync::Arc};

use crate::{
    device::{tensor::DenseMatrix, Device, OperationError},
    graph::{like::GraphLike, Graph, GraphNodeId, GraphNodeIdTy},
};

pub trait OptimiserState<D: Device>: Sized {
    type Params: Clone + Debug + Default;

    fn new(device: Arc<D>, size: usize, params: Self::Params) -> Result<Self, D::DeviceError>;

    fn update(
        &mut self,
        weights: &mut DenseMatrix<D>,
        grads: &mut DenseMatrix<D>,
        gradient_factor: f32,
        learning_rate: f32,
    ) -> Result<(), OperationError<D::DeviceError>>;

    fn reset(&mut self) -> Result<(), D::DeviceError>;

    fn load_from_checkpoint(
        map: &mut HashMap<String, &mut Self>,
        path: &str,
        old_format: bool,
    ) -> Result<(), OperationError<D::DeviceError>>;

    fn write_to_checkpoint(map: &HashMap<String, &Self>, path: &str) -> Result<(), D::DeviceError>;

    fn set_params(&mut self, params: Self::Params);
}

pub struct Optimiser<D: Device, G: GraphLike<D>, S: OptimiserState<D>> {
    phantom: PhantomData<D>,
    pub graph: G,
    pub state: HashMap<String, S>,
    pre_update: Vec<Box<dyn AdditionalUpdate<D>>>,
    post_update: Vec<Box<dyn AdditionalUpdate<D>>>,
}

pub trait AdditionalUpdate<D: Device> {
    fn apply_update(&mut self, graph: &mut Graph<D>) -> Result<(), OperationError<D::DeviceError>>;
}

impl<D: Device, G: GraphLike<D>, S: OptimiserState<D>> Optimiser<D, G, S> {
    pub fn new(graph: G, params: S::Params) -> Result<Self, D::DeviceError> {
        let weight_ids = graph.primary().weight_ids();

        let mut state = HashMap::new();

        for id in &weight_ids {
            let idx = graph.primary().weight_idx(id).unwrap();
            let w = graph.primary().get(GraphNodeId::new(idx, GraphNodeIdTy::Values)).unwrap();
            let w = w.dense();
            assert!(w.batch_size().is_none());
            let size = w.size();

            let single = S::new(graph.primary().device(), size, params.clone())?;

            let old = state.insert(id.clone(), single);
            assert!(old.is_none());
        }

        Ok(Self { phantom: PhantomData, graph, state, pre_update: Vec::new(), post_update: Vec::new() })
    }

    pub fn add_pre_update(&mut self, additional: impl AdditionalUpdate<D> + 'static) {
        self.pre_update.push(Box::new(additional));
    }

    pub fn add_post_update(&mut self, additional: impl AdditionalUpdate<D> + 'static) {
        self.post_update.push(Box::new(additional));
    }

    pub fn update(&mut self, gradient_factor: f32, learning_rate: f32) -> Result<(), OperationError<D::DeviceError>> {
        for additional in &mut self.pre_update {
            additional.apply_update(self.graph.primary_mut())?;
        }

        for id in &self.graph.primary().weight_ids() {
            let idx = self.graph.primary().weight_idx(id).unwrap();
            let weight_id = GraphNodeId::new(idx, GraphNodeIdTy::Values);

            let weights = &mut self.graph.primary().get(weight_id)?;
            let single = self.state.get_mut(id).unwrap();

            let grad_id = GraphNodeId::new(idx, GraphNodeIdTy::Gradients);
            if let Ok(grads) = self.graph.primary().get(grad_id) {
                self.graph.reduce_sum_into_first(&self.graph.get_all(grad_id)?)?;
                single.update(&mut *weights.dense_mut(), &mut *grads.dense_mut(), gradient_factor, learning_rate)?;
                self.graph.scatter_first_into_rest(&self.graph.get_all(weight_id)?)?;
            }
        }

        for additional in &mut self.post_update {
            additional.apply_update(self.graph.primary_mut())?;
        }

        Ok(())
    }

    pub fn reset_state(&mut self) -> Result<(), D::DeviceError> {
        for single in self.state.values_mut() {
            single.reset()?;
        }

        Ok(())
    }

    pub fn set_params_for_weight(&mut self, id: &str, params: S::Params) {
        self.state.get_mut(id).unwrap().set_params(params);
    }

    pub fn set_params(&mut self, params: S::Params) {
        for id in self.graph.primary().weight_ids() {
            self.set_params_for_weight(&id, params.clone());
        }
    }

    pub fn write_to_checkpoint(&self, path: &str) -> Result<(), D::DeviceError> {
        self.graph.primary().write_to_file(&format!("{path}/weights.bin"));
        let map = self.state.iter().map(|(id, single)| (id.clone(), single)).collect();
        S::write_to_checkpoint(&map, path)
    }

    pub fn load_from_checkpoint(&mut self, path: &str) -> Result<(), OperationError<D::DeviceError>> {
        self.load_from_checkpoint_(path, false)
    }

    pub fn load_weights_from_file(&mut self, path: &str) -> Result<(), OperationError<D::DeviceError>> {
        self.load_weights_from_file_(path, false)
    }

    pub fn load_from_old_format_checkpoint(&mut self, path: &str) -> Result<(), OperationError<D::DeviceError>> {
        self.load_from_checkpoint_(path, true)
    }

    pub fn load_old_format_weights_from_file(&mut self, path: &str) -> Result<(), OperationError<D::DeviceError>> {
        self.load_weights_from_file_(path, true)
    }

    fn load_weights_from_file_(&mut self, path: &str, old_format: bool) -> Result<(), OperationError<D::DeviceError>> {
        self.graph.primary_mut().load_from_file(path, old_format)?;

        let primary = self.graph.primary();

        for id in primary.weight_ids() {
            let idx = GraphNodeId::new(primary.weight_idx(&id).unwrap(), GraphNodeIdTy::Values);
            self.graph.scatter_first_into_rest(&self.graph.get_all(idx)?)?;
        }

        Ok(())
    }

    fn load_from_checkpoint_(&mut self, path: &str, old_format: bool) -> Result<(), OperationError<D::DeviceError>> {
        self.load_weights_from_file_(&format!("{path}/weights.bin"), old_format)?;
        let mut map = self.state.iter_mut().map(|(id, single)| (id.clone(), single)).collect();
        S::load_from_checkpoint(&mut map, path, old_format)
    }
}

pub struct WrapOptimiser<O, P> {
    optimiser: O,
    phantom_data: PhantomData<P>,
}

impl<D, O, P> OptimiserState<D> for WrapOptimiser<O, P>
where
    D: Device,
    O: OptimiserState<D>,
    P: Clone + Default + Debug + Into<O::Params>,
{
    type Params = P;

    fn new(device: Arc<D>, size: usize, params: Self::Params) -> Result<Self, D::DeviceError> {
        Ok(Self { optimiser: O::new(device, size, params.into())?, phantom_data: PhantomData })
    }

    fn update(
        &mut self,
        weights: &mut DenseMatrix<D>,
        grads: &mut DenseMatrix<D>,
        gradient_factor: f32,
        learning_rate: f32,
    ) -> Result<(), OperationError<D::DeviceError>> {
        self.optimiser.update(weights, grads, gradient_factor, learning_rate)
    }

    fn reset(&mut self) -> Result<(), D::DeviceError> {
        self.optimiser.reset()
    }

    fn set_params(&mut self, params: Self::Params) {
        self.optimiser.set_params(params.into());
    }

    fn load_from_checkpoint(
        map: &mut HashMap<String, &mut Self>,
        path: &str,
        old_format: bool,
    ) -> Result<(), OperationError<D::DeviceError>> {
        let mut map = map.iter_mut().map(|(id, single)| (id.clone(), &mut single.optimiser)).collect();
        O::load_from_checkpoint(&mut map, path, old_format)
    }

    fn write_to_checkpoint(map: &HashMap<String, &Self>, path: &str) -> Result<(), D::DeviceError> {
        let map = map.iter().map(|(id, single)| (id.clone(), &single.optimiser)).collect();
        O::write_to_checkpoint(&map, path)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::device::{cpu::CpuThread, tensor::DenseMatrix};

    use super::{
        ranger21::{Ranger21, Ranger21Params},
        OptimiserState,
    };

    fn dense_values(weights: &DenseMatrix<CpuThread>) -> Vec<f32> {
        let mut values = vec![0.0; weights.size()];
        weights.write_to_slice(&mut values).unwrap();
        values
    }

    fn reference_ranger21_update(
        weight: f32,
        grad: f32,
        step: usize,
        learning_rate: f32,
        params: Ranger21Params,
    ) -> f32 {
        let beta1_sq = params.beta1 * params.beta1;
        let momentum = (1.0 - beta1_sq) * grad;
        let velocity = (1.0 - params.beta2) * grad * grad;
        let bias_correction1 = 1.0 - params.beta1.powi(step as i32);
        let bias_correction2 = 1.0 - params.beta2.powi(step as i32);
        let noise_norm = ((1.0f32 + params.beta2).powi(2) + params.beta2.powi(2)).sqrt();
        let denom = (velocity / bias_correction2).sqrt() + params.eps;
        weight - learning_rate * momentum / (bias_correction1 * noise_norm * denom)
    }

    fn reference_ranger21_pnm_zero_update(
        weight: f32,
        grad: f32,
        step: usize,
        learning_rate: f32,
        params: Ranger21Params,
        momentum: &mut f32,
        velocity: &mut f32,
    ) -> f32 {
        let beta1_sq = params.beta1 * params.beta1;
        *momentum = beta1_sq * *momentum + (1.0 - beta1_sq) * grad;
        *velocity = params.beta2 * *velocity + (1.0 - params.beta2) * grad * grad;

        let bias_correction1 = 1.0 - params.beta1.powi(step as i32);
        let bias_correction2 = 1.0 - params.beta2.powi(step as i32);
        let noise_norm = ((1.0f32 + params.beta2).powi(2) + params.beta2.powi(2)).sqrt();
        let denom = (*velocity / bias_correction2).sqrt() + params.eps;
        weight - learning_rate * *momentum / (bias_correction1 * noise_norm * denom)
    }

    #[test]
    fn ranger21_first_step_matches_nnue_pytorch_adamw_pnm_zero() {
        let device = Arc::new(CpuThread);
        let mut weights = DenseMatrix::zeroed(device.clone(), 2, None).unwrap();
        weights.load_from_slice(None, &[1.0, -2.0]).unwrap();
        let mut grads = DenseMatrix::zeroed(device.clone(), 2, None).unwrap();
        grads.load_from_slice(None, &[0.5, -0.25]).unwrap();

        let params = Ranger21Params { clip: None, ..Default::default() };
        let mut optimiser = Ranger21::<CpuThread>::new(device, 2, params).unwrap();
        optimiser.update(&mut weights, &mut grads, 1.0, 0.001).unwrap();

        let actual = dense_values(&weights);
        let expected = [
            reference_ranger21_update(1.0, 0.5, 1, 0.001, params),
            reference_ranger21_update(-2.0, -0.25, 1, 0.001, params),
        ];

        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() < 1.0e-7, "{actual} != {expected}");
        }
    }

    #[test]
    fn ranger21_lookahead_cache_starts_from_initial_weights() {
        let device = Arc::new(CpuThread);
        let mut weights = DenseMatrix::zeroed(device.clone(), 2, None).unwrap();
        weights.load_from_slice(None, &[1.0, -2.0]).unwrap();
        let mut grads = DenseMatrix::zeroed(device.clone(), 2, None).unwrap();
        grads.load_from_slice(None, &[0.0, 0.0]).unwrap();

        let params = Ranger21Params { clip: None, ..Default::default() };
        let mut optimiser = Ranger21::<CpuThread>::new(device, 2, params).unwrap();
        for _ in 0..params.lookahead_mergetime {
            optimiser.update(&mut weights, &mut grads, 1.0, 0.001).unwrap();
        }

        assert_eq!(dense_values(&weights), vec![1.0, -2.0]);
    }

    #[test]
    fn ranger21_second_step_uses_alternating_pnm_buffers() {
        let device = Arc::new(CpuThread);
        let mut weights = DenseMatrix::zeroed(device.clone(), 1, None).unwrap();
        weights.load_from_slice(None, &[1.0]).unwrap();
        let mut grads = DenseMatrix::zeroed(device.clone(), 1, None).unwrap();
        grads.load_from_slice(None, &[0.5]).unwrap();

        let params = Ranger21Params { clip: None, ..Default::default() };
        let mut optimiser = Ranger21::<CpuThread>::new(device, 1, params).unwrap();
        optimiser.update(&mut weights, &mut grads, 1.0, 0.001).unwrap();
        optimiser.update(&mut weights, &mut grads, 1.0, 0.001).unwrap();

        let mut pos_momentum = 0.0;
        let mut neg_momentum = 0.0;
        let mut velocity = 0.0;
        let weight_after_step1 =
            reference_ranger21_pnm_zero_update(1.0, 0.5, 1, 0.001, params, &mut pos_momentum, &mut velocity);
        let expected = reference_ranger21_pnm_zero_update(
            weight_after_step1,
            0.5,
            2,
            0.001,
            params,
            &mut neg_momentum,
            &mut velocity,
        );
        let actual = dense_values(&weights)[0];

        assert!((actual - expected).abs() < 1.0e-7, "{actual} != {expected}");
    }
}
