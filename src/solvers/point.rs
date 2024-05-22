use std::collections::HashMap;

use ndarray::Zip;

use super::{solver::SolverOps, SolverInput, SolverState};
use crate::{chem::Point, domain::Mesh, fields::RField};

#[derive(Debug)]
pub struct PointSolver {
    point: Point,
    state: SolverState,
}

impl PointSolver {
    pub fn new(point: Point, mesh: Mesh) -> Self {
        let state = SolverState::new(mesh, &[point.monomer_id()]);
        Self { point, state }
    }
}

impl SolverOps for PointSolver {
    fn state(&self) -> &SolverState {
        &self.state
    }

    fn update_state<'a>(&mut self, input: &SolverInput<'a>) {
        let monomer_id = self.point.monomer_id();

        let monomer = input.monomers[monomer_id];
        let field = &input.fields[monomer_id];

        // Field is created upon object construction
        let mut density = self.state.density.get_mut(&monomer_id).unwrap();

        // Compute density field and partition function from omega field
        self.state.partition = 0.0;
        Zip::from(density.view_mut())
            .and(field)
            .for_each(|rho, omega| {
                *rho = (-monomer.size * omega).exp();
                self.state.partition += *rho;
            });

        // Normalize partition sum
        self.state.partition /= field.len() as f64;

        // Normalize density inplace
        density.mapv_inplace(|rho| rho / self.state.partition);
    }
}
