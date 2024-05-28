use crate::{system::System, Result};

pub trait FieldUpdater {
    fn step(&mut self, system: &mut System) -> Result<()>;

    fn is_converged(&self, _system: &System) -> bool {
        false
    }
}
