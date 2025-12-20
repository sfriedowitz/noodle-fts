use std::collections::HashMap;

use super::SolverOps;
use crate::{
    chem::{Point, Species, SpeciesDescription},
    domain::Mesh,
    fields::RField,
};

#[derive(Debug)]
pub struct PointSolver {
    mesh: Mesh,
    species: Point,
    concentrations: HashMap<usize, RField>,
    partition: f64,
}

impl PointSolver {
    pub fn new(mesh: Mesh, species: Point) -> Self {
        let concentrations = HashMap::from([(species.monomer.id, RField::zeros(mesh))]);
        Self {
            mesh,
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

    fn solve(&mut self, fields: &HashMap<usize, RField>, _ksq: &RField) {
        let monomer = self.species.monomer;
        let field = &fields.get(&monomer.id).unwrap();
        let concentration = self.concentrations.get_mut(&monomer.id).unwrap();

        // Compute concentration field and partition function from omega field
        let mut partition_sum = 0.0;
        concentration.zip_mut_with(field, |c, w| {
            *c = (-monomer.size * w).exp();
            partition_sum += *c;
        });

        // Normalize partition sum
        self.partition = partition_sum / self.mesh.size() as f64;

        // Normalize concentration inplace
        let prefactor = self.species.phi() / self.partition;
        concentration.mapv_inplace(|conc| prefactor * conc);
    }

    fn stress(&mut self, domain: &crate::domain::Domain) -> Vec<f64> {
        let volume = domain.cell().volume();

        // Point particles contribute only translational entropy stress
        // σ_trans = -(φ/V) δ_ij (diagonal, isotropic pressure)
        let phi = self.species.phi();
        let pressure = -phi / volume;

        let ncomponents = domain.mesh().stress_components();
        let mut stress = vec![pressure; ncomponents];

        // Set off-diagonal components to zero
        match domain.mesh() {
            crate::domain::Mesh::One(_) => {}                  // No off-diagonal terms
            crate::domain::Mesh::Two(_, _) => stress[2] = 0.0, // σ_xy = 0
            crate::domain::Mesh::Three(_, _, _) => {
                stress[3] = 0.0; // σ_xy = 0
                stress[4] = 0.0; // σ_xz = 0
                stress[5] = 0.0; // σ_yz = 0
            }
        }

        stress
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::{
        chem::Monomer,
        domain::{Domain, UnitCell},
    };

    #[test]
    fn test_zero_field() {
        let mesh = Mesh::One(16);
        let cell = UnitCell::lamellar(1.0).unwrap();
        let domain = Domain::new(mesh, cell).unwrap();

        let phi = 0.235;
        let point = Point::new(Monomer::new(0, 1.0), phi);
        let fields = [(0, RField::zeros(mesh))].into();
        let ksq = domain.ksq();

        let mut solver = PointSolver::new(mesh, point);
        solver.solve(&fields, &ksq);

        // Concentration should be the bulk fraction in the absence of a field
        let conc = solver.concentrations().get(&point.monomer.id).unwrap();
        assert!(conc.iter().all(|elem| *elem == phi));
    }
}
