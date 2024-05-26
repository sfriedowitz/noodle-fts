use crate::system::System;

pub trait FieldUpdater {
    fn step(&mut self, system: &mut System);
}
