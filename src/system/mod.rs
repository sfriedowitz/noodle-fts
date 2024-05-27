mod interaction;
mod system;

pub use interaction::Interaction;
pub use system::System;

#[derive(thiserror::Error, Debug)]
pub enum SystemError {
    #[error("domain missing from system builder")]
    MissingDomain,
    #[error("interaction missing from system builder")]
    MissingInteraction,
    #[error("system must contain at least one species")]
    MissingSpecies,
    #[error("monomer IDs must be consecutive from [0, nmonomer)")]
    MonomerIDs,
}
