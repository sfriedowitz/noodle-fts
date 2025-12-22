pub mod cell;
pub mod field;

pub use cell::CellUpdater;
pub use field::FieldUpdater;

use crate::system::System;

pub trait SystemUpdater {
    /// Perform a single update step on the system
    fn step(&mut self, system: &mut System) -> crate::Result<()>;
}
