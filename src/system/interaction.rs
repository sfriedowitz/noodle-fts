use ndarray::Array2;

use crate::fields::RField;

pub struct Interaction {
    chi: Array2<f64>,
}

impl Interaction {
    pub fn new(nmonomer: usize) -> Self {
        let chi = Array2::zeros((nmonomer, nmonomer));
        Self { chi }
    }

    pub fn set_chi(&mut self, i: usize, j: usize, chi: f64) {
        todo!()
    }

    pub fn energy(&self, density: &[RField]) -> f64 {
        todo!()
    }

    pub fn energy_bulk(&self, density: &[f64]) -> f64 {
        todo!()
    }

    pub fn gradient(&self, id: usize, density: &[RField], field: &mut RField) {
        todo!()
    }
}
