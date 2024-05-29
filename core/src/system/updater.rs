use ndarray::Zip;

use crate::{system::System, RField};

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
    temp_fields: Vec<RField>,
}

impl FieldUpdater {
    pub fn new(system: &System, delta: f64, stopping_tol: Option<f64>) -> Self {
        let temp_fields = system.fields().to_vec();
        Self {
            delta,
            stopping_tol,
            temp_fields,
        }
    }

    pub fn is_converged(&self, system: &System) -> bool {
        self.stopping_tol
            .map(|tol| system.field_error() <= tol)
            .unwrap_or(false)
    }

    pub fn step(&mut self, system: &mut System) {
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

        // 4) Evaluate
        system.update();
    }
}
