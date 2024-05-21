use ndarray::Zip;

use super::solver::{SolverInput, SpeciesSolver};
use crate::{
    chem::{monomer::Monomer, species::Species},
    domain::RField,
};

pub struct PointSolver {
    density: RField,
}

impl SpeciesSolver for PointSolver {
    fn solve<'a>(&mut self, input: SolverInput<'a>) {
        let monomer = input.monomers[self.monomer_id];
        let omega = &input.omegas[self.monomer_id];

        // Compute density field and partition function from omega field
        self.q = 0.0;
        Zip::from(&mut self.density)
            .and(omega)
            .for_each(|rho, omega| {
                *rho = (-monomer.size * omega).exp();
                self.q += *rho;
            });
        self.q /= input.domain.mesh_size() as f64;

        // Normalize density field
        self.density.mapv_inplace(|rho| rho / self.q);
    }
}
