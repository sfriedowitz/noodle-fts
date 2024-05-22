use crate::{
    domain::{Mesh, FFT},
    fields::{CField, RField},
};

#[derive(Debug, Clone, Copy)]
pub enum StepMethod {
    RK2,
    RQM4,
}

#[derive(Debug)]
pub struct BlockPropagator {
    fft: FFT,
    // Operator arrays
    lw1: RField,
    lw2: RField,
    lk1: RField,
    lk2: RField,
    // Work arrays
    q1: RField,
    q2: RField,
    qr: RField,
    qk: CField,
}

impl BlockPropagator {
    pub fn new(mesh: Mesh) -> Self {
        todo!()
    }

    pub fn update_operators(&mut self) {
        todo!()
    }
}
