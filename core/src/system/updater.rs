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
    delta: f64,
    stopping_tol: Option<f64>,
    buffers: Vec<RField>,
}

impl FieldUpdater {
    pub fn new(system: &System, delta: f64, stopping_tol: Option<f64>) -> Self {
        let buffers = system.fields().to_vec();
        Self {
            delta,
            stopping_tol,
            buffers,
        }
    }

    pub fn is_converged(&self, system: &System) -> bool {
        self.stopping_tol
            .map(|tol| system.field_error() <= tol)
            .unwrap_or(false)
    }

    pub fn step(&mut self, system: &mut System) {
        // 1) Predict
        for (state, buffer) in system.iter_mut().zip(self.buffers.iter_mut()) {
            Zip::from(state.field)
                .and(buffer)
                .and(state.residual)
                .for_each(|w, b, r| {
                    *w += self.delta * r;
                    *b = *w + 0.5 * self.delta * r;
                })
        }

        // 2) Evaluate
        system.update();

        // 3) Correct
        for (state, buffer) in system.iter().zip(self.buffers.iter_mut()) {
            buffer.zip_mut_with(state.residual, |b, r| *b += 0.5 * self.delta * r);
        }
        system.assign_fields(&self.buffers).unwrap();

        // 4) Evaluate
        system.update();
    }
}
