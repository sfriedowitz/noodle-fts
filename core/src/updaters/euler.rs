use ndarray::Zip;

use super::FieldUpdater;
use crate::{system::System, RField};

pub struct EulerUpdater {
    delta: f64,
    temp_fields: Vec<RField>,
}

impl EulerUpdater {
    pub fn new(system: &System, delta: f64) -> Self {
        Self {
            delta,
            temp_fields: system.fields().to_vec(),
        }
    }
}

impl FieldUpdater for EulerUpdater {
    fn step(&mut self, system: &mut System) {
        // 1) Predict
        for (state, temp) in system.iter_mut().zip(self.temp_fields.iter_mut()) {
            Zip::from(state.field)
                .and(state.residual)
                .and(temp)
                .for_each(|w, r, t| {
                    *w += self.delta * r;
                    *t = *w + 0.5 * self.delta * r;
                })
        }

        // 2) Evaluate
        system.update();

        // 3) Correct
        for (state, temp) in system.iter().zip(self.temp_fields.iter_mut()) {
            Zip::from(state.residual).and(temp).for_each(|r, t| {
                *t += 0.5 * self.delta * r;
            })
        }
        system.assign_fields(&self.temp_fields).unwrap();

        // // 4) Evaluate
        system.update();
    }
}
