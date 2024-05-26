use ndarray::Zip;

use super::{SolverOps, SolverState};
use crate::{
    chem::{Point, Species},
    domain::{Domain, Mesh},
    RField,
};

#[derive(Debug)]
pub struct PointSolver {
    point: Point,
    state: SolverState,
}

impl PointSolver {
    pub fn new(mesh: Mesh, point: Point) -> Self {
        let state = SolverState::new(mesh, [point.monomer]);
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

    fn solve(&mut self, domain: &Domain, fields: &[RField]) {
        let monomer = self.point.monomer;
        let field = &fields[monomer.id];
        let density = self.state.density.get_mut(&monomer.id).unwrap();

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
