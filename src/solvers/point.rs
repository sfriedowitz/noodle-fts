use std::collections::HashMap;

use ndarray::Zip;

use super::{SolverOps, SolverState};
use crate::{
    chem::{Monomer, Point, Species, SpeciesDescription},
    domain::{Domain, Mesh},
    fields::RField,
};

#[derive(Debug)]
pub struct PointSolver {
    point: Point,
    state: SolverState,
}

impl PointSolver {
    pub fn new(point: Point, mesh: Mesh) -> Self {
        let mut state = SolverState::default();
        state.density.insert(point.monomer.id, RField::zeros(mesh));
        Self { point, state }
    }
}

impl SolverOps for PointSolver {
    fn species(&self) -> Species {
        self.point.into()
    }

    fn state(&self) -> &SolverState {
        &self.state
    }

    fn solve<'a>(&mut self, domain: &Domain, fields: &[RField], monomers: &[Monomer]) {
        let monomer_id = self.point.monomer.id;
        let monomer = monomers[monomer_id];
        let field = &fields[monomer_id];
        let mut density = self.state.density.get_mut(&monomer_id).unwrap();

        // Compute density field and partition function from omega field
        let mut partition_sum = 0.0;
        Zip::from(density.view_mut())
            .and(field)
            .for_each(|rho, omega| {
                *rho = (-monomer.size * omega).exp();
                partition_sum += *rho;
            });

        // Normalize partition sum
        self.state.partition = partition_sum / domain.mesh_size() as f64;

        // Normalize density inplace
        let prefactor = self.point.fraction / self.state.partition;
        density.mapv_inplace(|rho| prefactor * rho);
    }
}
