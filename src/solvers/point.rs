use std::{borrow::BorrowMut, collections::HashMap};

use ndarray::Zip;

use super::{
    solver::{SolverOps, SpeciesSolver},
    SolverInput, SolverState,
};
use crate::{
    chem::Point,
    domain::Mesh,
    error::{FTSError, Result},
    fields::RField,
};

#[derive(Debug)]
pub struct PointSolver {
    point: Point,
    state: SolverState,
}

impl PointSolver {
    pub fn new(point: Point, mesh: Mesh) -> Self {
        let mut state = SolverState {
            partition: 0.0,
            density: HashMap::from([(point.monomer_id(), RField::zeros(mesh))]),
        };
        Self { point, state }
    }
}

impl SolverOps for PointSolver {
    fn state(&self) -> &SolverState {
        &self.state
    }

    fn update_state<'a>(&mut self, input: &SolverInput<'a>) -> Result<()> {
        let monomer_id = self.point.monomer_id();

        let monomer = input.monomers[monomer_id];
        let field = &input.fields[monomer_id];

        if let Some(density) = self.state.density.get_mut(&monomer_id) {
            // Compute density field and partition function from omega field
            let mut partition = 0.0;
            Zip::from(density.view_mut())
                .and(field)
                .for_each(|rho, omega| {
                    *rho = (-monomer.size * omega).exp();
                    partition += *rho;
                });

            // Normalize partition sum
            self.state.partition = partition / field.len() as f64;

            // Normalize density inplace
            density.mapv_inplace(|rho| rho / self.state.partition);

            Ok(())
        } else {
            Err(FTSError::Generic(format!(
                "monomer {monomer_id} density missing from species state"
            )))
        }
    }
}
