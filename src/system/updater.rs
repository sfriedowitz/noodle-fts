use ndarray::Zip;

use crate::{fields::RField, system::System};

/// Field updater implementing a Euler-Maruyama predictor-corrector method.
///
/// The updater uses a "PECE" (predictor-evaluate-correct-evaluate) scheme,
/// which is empirically found to be stable up to larger timesteps
/// at the cost of an additional system evaluation.
///
/// Reference: https://en.wikipedia.org/wiki/Predictor-corrector_method
#[derive(Debug, Clone)]
pub struct FieldUpdater {
    dt: f64,
    buffers: Vec<RField>,
}

impl FieldUpdater {
    pub fn new(system: &System, dt: f64) -> Self {
        Self {
            dt,
            buffers: system.fields().to_vec(),
        }
    }

    pub fn step(&mut self, system: &mut System) {
        // 1) Predict with the "force" at time t
        for (state, buffer) in system.iter_mut().zip(self.buffers.iter_mut()) {
            Zip::from(state.field)
                .and(buffer)
                .and(state.residual)
                .for_each(|w, b, r| {
                    *w += self.dt * r;
                    *b = *w + 0.5 * self.dt * r;
                })
        }

        // 2) Evaluate with the predicted fields
        system.update();

        // 3) Correct by averaging the "force" at time (t, t+1)
        for (state, buffer) in system.iter().zip(self.buffers.iter_mut()) {
            buffer.zip_mut_with(state.residual, |b, r| *b += 0.5 * self.dt * r);
        }

        // 4) Evaluate the final "force" at time t+1
        system.assign_fields(&self.buffers).unwrap();
        system.update();
    }
}
