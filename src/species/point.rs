use ndarray::Zip;

use crate::field::RField;

pub struct Point {
    q: f64,
    mu: f64,
    phi: f64,
    rho: RField,
}

impl Point {
    pub fn solve(&mut self, omega: &RField) {
        let vol = 1.0; // TODO: Update this
        let mesh_size = 32 * 32 * 32;

        // Compute density field and partition function from omega field
        self.q = 0.0;
        Zip::from(&mut self.rho).and(omega).for_each(|rho, omega| {
            *rho = (-vol * omega).exp();
            self.q += *rho;
        });
        self.q /= mesh_size as f64;

        // Normalize density field
        self.rho.mapv_inplace(|rho| rho / self.q);
    }
}
