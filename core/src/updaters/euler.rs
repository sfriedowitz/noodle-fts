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

        // // 4) Evaluate
        system.update();
    }
}
