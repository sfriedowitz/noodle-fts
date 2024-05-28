use crate::system::System;

pub trait FieldUpdater {
    fn step(&mut self, system: &mut System);

    fn is_converged(&self, _system: &System) -> bool {
        false
    }
}
