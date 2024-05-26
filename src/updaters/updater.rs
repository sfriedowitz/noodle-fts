use crate::system::System;

pub trait FieldUpdater {
    fn is_converged(&self, system: &System);

    fn step(&mut self, system: &mut System);
}
