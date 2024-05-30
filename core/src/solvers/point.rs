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
    concentrations: HashMap<usize, RField>,
    partition: f64,
}

impl PointSolver {
    pub fn new(mesh: Mesh, species: Point) -> Self {
        let concentrations = HashMap::from([(species.monomer.id, RField::zeros(mesh))]);
        Self {
            species,
            concentrations,
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

    fn concentrations(&self) -> &HashMap<usize, RField> {
        &self.concentrations
    }

    fn solve(&mut self, domain: &Domain, fields: &[RField]) {
        let monomer = self.species.monomer;
        let field = &fields[monomer.id];
        let concentration = self.concentrations.get_mut(&monomer.id).unwrap();

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

#[cfg(test)]
mod tests {

    use super::*;
    use crate::{chem::Monomer, domain::UnitCell};

    #[test]
    fn test_zero_field() {
        let mesh = Mesh::One(16);
        let cell = UnitCell::lamellar(1.0).unwrap();
        let domain = Domain::new(mesh, cell).unwrap();

        let phi = 0.235;
        let point = Point::new(Monomer::new(0, 1.0), phi);
        let fields = vec![RField::zeros(mesh); 1];

        let mut solver = PointSolver::new(mesh, point);
        solver.solve(&domain, &fields);

        // Concentration should be the bulk fraction in the absence of a field
        let conc = solver.concentrations().get(&point.monomer.id).unwrap();
        assert!(conc.iter().all(|elem| *elem == phi));
    }
}
