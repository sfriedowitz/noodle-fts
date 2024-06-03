use std::collections::HashMap;

use ndarray::Zip;

use crate::{fields::RField, system::System, Result};

/// Field updater implementing a Euler-Maruyama predictor-corrector method.
///
/// The updater uses a "PECE" (predictor-evaluate-correct-evaluate) scheme,
/// which is empirically found to be stable up to larger timesteps
/// at the cost of an additional system evaluation.
///
/// Reference: https://en.wikipedia.org/wiki/Predictor-corrector_method
#[derive(Debug, Clone)]
pub struct FieldUpdater {
    step_size: f64,
    buffers: HashMap<usize, RField>,
}

impl FieldUpdater {
    pub fn new(system: &System, step_size: f64) -> Self {
        Self {
            step_size,
            buffers: system.fields().clone(),
        }
    }

    fn get_buffer(&mut self, id: usize) -> Result<&mut RField> {
        self.buffers
            .get_mut(&id)
            .ok_or(format!("monomer {id} is not present in field updater").into())
    }

    pub fn step(&mut self, system: &mut System) -> Result<()> {
        let step_size = self.step_size;

        // 1) Predict with the "force" at time t
        for state in system.iter_mut() {
            let buffer = self.get_buffer(state.id)?;
            Zip::from(state.field)
                .and(buffer)
                .and(state.residual)
                .for_each(|w, b, r| {
                    *w += step_size * r;
                    *b = *w + 0.5 * step_size * r;
                })
        }

        // 2) Evaluate with the predicted fields
        system.update();

        // 3) Correct by averaging the "force" at time (t, t+1)
        for state in system.iter() {
            let buffer = self.get_buffer(state.id)?;
            buffer.zip_mut_with(state.residual, |b, r| *b += 0.5 * step_size * r);
        }

        // 4) Evaluate the final "force" at time t+1
        for (id, buffer) in self.buffers.iter() {
            system.assign_field(*id, buffer.view()).unwrap();
        }
        system.update();

        Ok(())
    }
}
