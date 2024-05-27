use std::collections::HashMap;

use ndarray::Zip;

use super::SolverOps;
use crate::{
    chem::{Point, Species, SpeciesDescription},
    domain::{Domain, Mesh},
    RField,
};

#[derive(Debug)]
pub struct PointSolver {
    species: Point,
    density: HashMap<usize, RField>,
    partition: f64,
}

impl PointSolver {
    pub fn new(mesh: Mesh, species: Point) -> Self {
        let density = HashMap::from([(species.monomer.id, RField::zeros(mesh))]);
        Self {
            species,
            density,
            partition: 1.0,
        }
    }
}

impl SolverOps for PointSolver {
    fn species(&self) -> Species {
        self.species.into()
    }

    fn partition(&self) -> f64 {
        self.partition
    }

    fn density(&self) -> &HashMap<usize, RField> {
        &self.density
    }

    fn solve(&mut self, domain: &Domain, fields: &[RField]) {
        let monomer = self.species.monomer;
        let field = &fields[monomer.id];
        let density = self.density.get_mut(&monomer.id).unwrap();

        // Compute density field and partition function from omega field
        let mut partition_sum = 0.0;
        Zip::from(density.view_mut())
            .and(field)
            .for_each(|rho, omega| {
                *rho = (-monomer.size * omega).exp();
                partition_sum += *rho;
            });

        // Normalize partition sum
        self.partition = partition_sum / domain.mesh_size() as f64;

        // Normalize density inplace
        let prefactor = self.species.phi() / self.partition;
        density.mapv_inplace(|rho| prefactor * rho);
    }
}
