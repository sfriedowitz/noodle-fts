mod interaction;
mod system;

pub use interaction::Interaction;
pub use system::System;

#[derive(thiserror::Error, Debug)]
pub enum SystemError {
    #[error("number of fields != number of monomers")]
    NumFields,
    #[error("validation error: {0}")]
    Validation(String),
}
