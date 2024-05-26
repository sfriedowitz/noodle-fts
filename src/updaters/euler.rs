use super::FieldUpdater;
use crate::{system::System, RField};

pub struct EulerUpdater {
    delta: f64,
    temp_fields: Vec<RField>,
}

impl FieldUpdater for EulerUpdater {
    fn is_converged(&self, system: &System) {
        todo!()
    }

    fn step(&mut self, system: &mut System) {
        todo!()
    }
}
