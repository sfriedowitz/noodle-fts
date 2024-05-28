use ndarray::Zip;

use super::FieldUpdater;
use crate::{system::System, RField};

pub struct EulerUpdater {
    delta: f64,
    fields_temp: Vec<RField>,
}

impl EulerUpdater {
    pub fn new(system: &System, delta: f64) -> Self {
        Self {
            delta,
            fields_temp: system.fields().to_vec(),
        }
    }
}

impl FieldUpdater for EulerUpdater {
    fn step(&mut self, system: &mut System) {
        // 1) Predict
        for ((field, residual), temp) in system.iter_updater().zip(&mut self.fields_temp) {
            Zip::from(field)
                .and(residual)
                .and(temp)
                .for_each(|w, r, t| {
                    *w += self.delta * r;
                    *t = *w + 0.5 * self.delta * r;
                })
        }

        // 2) Evaluate
        system.update();

        // 3) Correct
        for (residual, temp) in system.residuals().iter().zip(&mut self.fields_temp) {
            Zip::from(residual).and(temp).for_each(|r, t| {
                *t += 0.5 * self.delta * r;
            })
        }
        system.assign_fields(&self.fields_temp).unwrap();

        // 4) Evaluate
        system.update();
    }
}

#[cfg(test)]
mod tests {

    use ndarray_rand::rand_distr::Normal;

    use super::*;
    use crate::{
        chem::{Block, Monomer, Polymer, Species},
        domain::{Domain, Mesh, UnitCell},
        system::Interaction,
    };

    #[test]
    fn test_updater() {
        let mesh = Mesh::One(32);
        let cell = UnitCell::lamellar(10.0).unwrap();
        let domain = Domain::new(mesh, cell).unwrap();

        let monomer_a = Monomer::new(0, 1.0);
        let monomer_b = Monomer::new(1, 1.0);

        let block_a = Block::new(monomer_a, 50, 1.0);
        let block_b = Block::new(monomer_b, 50, 1.0);

        let polymer = Polymer::new(vec![block_a, block_b], 100, 1.0);
        let species: Vec<Species> = vec![polymer.into()];

        let mut itx = Interaction::new(2);
        itx.set_chi(0, 1, 0.0);

        // When: System initialized with zero fields (density is equal to bulk values)
        let mut system = System::new(domain, itx, species).unwrap();
        let mut updater = EulerUpdater::new(&system, 0.1);

        let distr = Normal::new(0.0, 0.1).unwrap();
        let mut rng = rand::thread_rng();
        system.sample_fields(&distr, &mut rng);

        for _ in 0..100 {
            updater.step(&mut system);
            dbg!(system.free_energy());
        }

        dbg!(system.fields());
    }
}
