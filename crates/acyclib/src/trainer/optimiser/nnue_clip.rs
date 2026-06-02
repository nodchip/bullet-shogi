use crate::{
    device::{Device, OperationError},
    graph::Graph,
};

use super::AdditionalUpdate;

#[derive(Clone, Debug)]
pub struct NnuePytorchClipping {
    l1_rows: usize,
    l1_cols: usize,
    l1_offset_rows: usize,
    l1_offset_cols: usize,
    hidden_min: f32,
    hidden_max: f32,
    output_min: f32,
    output_max: f32,
}

impl NnuePytorchClipping {
    pub fn new(
        l1_rows: usize,
        l1_cols: usize,
        l1_offset_rows: usize,
        l1_offset_cols: usize,
        hidden_clip: f32,
        output_clip: f32,
    ) -> Self {
        Self {
            l1_rows,
            l1_cols,
            l1_offset_rows,
            l1_offset_cols,
            hidden_min: -hidden_clip,
            hidden_max: hidden_clip,
            output_min: -output_clip,
            output_max: output_clip,
        }
    }
}

impl<D: Device> AdditionalUpdate<D> for NnuePytorchClipping {
    fn apply_update(&mut self, graph: &mut Graph<D>) -> Result<(), OperationError<D::DeviceError>> {
        let l1fw = graph.get_weights("l1fw");
        let l1w = graph.get_weights("l1w");
        l1w.dense_mut().clip_with_repeated_offset(
            self.l1_rows,
            self.l1_cols,
            &l1fw.dense(),
            self.l1_offset_rows,
            self.l1_offset_cols,
            self.hidden_min,
            self.hidden_max,
        )?;

        graph.get_weights("l2w").dense_mut().clamp(self.hidden_min, self.hidden_max)?;
        graph.get_weights("l3w").dense_mut().clamp(self.output_min, self.output_max)?;

        Ok(())
    }
}
