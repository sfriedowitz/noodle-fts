use super::FieldUpdater;
use crate::system::System;

pub struct EulerUpdater {}

impl FieldUpdater for EulerUpdater {
    fn step(&mut self, system: &mut System) {
        todo!()
    }
}
