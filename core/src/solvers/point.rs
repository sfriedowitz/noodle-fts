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
    concentration: HashMap<usize, RField>,
    partition: f64,
}

impl PointSolver {
    pub fn new(mesh: Mesh, species: Point) -> Self {
        let concentration = HashMap::from([(species.monomer.id, RField::zeros(mesh))]);
        Self {
            species,
            concentration,
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

    fn concentration(&self) -> &HashMap<usize, RField> {
        &self.concentration
    }

    fn solve(&mut self, domain: &Domain, fields: &[RField]) {
        let monomer = self.species.monomer;
        let field = &fields[monomer.id];
        let concentration = self.concentration.get_mut(&monomer.id).unwrap();

        // Compute concentration field and partition function from omega field
        let mut partition_sum = 0.0;
        Zip::from(concentration.view_mut())
            .and(field)
            .for_each(|conc, field| {
                *conc = (-monomer.size * field).exp();
                partition_sum += *conc;
            });

        // Normalize partition sum
        self.partition = partition_sum / domain.mesh_size() as f64;

        // Normalize concentration inplace
        let prefactor = self.species.phi() / self.partition;
        concentration.mapv_inplace(|conc| prefactor * conc);
    }
}
