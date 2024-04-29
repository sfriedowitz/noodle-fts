use crate::{fft::FFT, mesh::Mesh, types::RField};

pub enum StepMethod {
    RK2,
    RQM4,
}

pub struct MDESolver {
    method: StepMethod,
    lw1: RField,
    lw2: RField,
    ld1: RField,
    ld2: RField,
}

impl MDESolver {
    pub fn new(mesh: &Mesh, method: StepMethod) -> Self {
        todo!()
    }

    pub fn update(&mut self, omega: RField, ksq: RField, vol: f64, b: f64, ds: f64) {
        todo!()
    }

    pub fn propagate(&mut self, fft: &mut FFT, q_in: &RField, q_out: &mut RField) {
        todo!()
    }
}
