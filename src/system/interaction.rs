use ndarray::Array2;

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
}
