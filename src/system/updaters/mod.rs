pub mod cell;
pub mod field;

pub use cell::CellUpdater;
pub use field::FieldUpdater;

use crate::{Result, system::System};

pub trait SystemUpdater {
    fn step(&mut self, system: &mut System) -> Result<()>;
}
