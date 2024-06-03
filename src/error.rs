use thiserror::Error;

pub type Result<T> = core::result::Result<T, Error>;

#[derive(Error, Debug)]
pub enum Error {
    #[error("unknown ID: monomer {0}")]
    UnknownId(usize),
    #[error("shape mismatch: {0:?} != {1:?}")]
    Shape(Vec<usize>, Vec<usize>),
    #[error("dimension mismatch: {0} != {1}")]
    Dimension(usize, usize),
    #[error("{0}")]
    Generic(#[from] Box<dyn std::error::Error>),
}
