mod interaction;
mod system;

pub use interaction::Interaction;
pub use system::System;

#[derive(thiserror::Error, Debug)]
pub enum SystemError {
    #[error("system must contain at least one species")]
    EmptySpecies,
    #[error("number of monomers does not match system")]
    NumMonomers,
    #[error("non-consecutive monomer IDs: {0:?}")]
    NonConsecutiveIDs(Vec<usize>),
}
