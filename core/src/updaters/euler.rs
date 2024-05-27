use super::FieldUpdater;
use crate::{system::System, RField};

pub struct EulerUpdater {
    delta: f64,
    fields_temp: Vec<RField>,
}

impl FieldUpdater for EulerUpdater {
    fn step(&mut self, system: &mut System) {
        todo!()
    }

    fn is_converged(&self, system: &System) {
        todo!()
    }
}
