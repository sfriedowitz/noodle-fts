use thiserror::Error;

pub type Result<T> = core::result::Result<T, Error>;

#[derive(Error, Debug)]
pub enum Error {
    // Custom
    #[error("monomer {0} is not present in the system")]
    MissingId(usize),
    #[error("{0}")]
    Generic(String),
    // External
    #[error("{0}")]
    Linalg(#[from] ndarray_linalg::error::LinalgError),
    #[error("{0}")]
    Shape(#[from] ndarray::ShapeError),
}

impl From<String> for Error {
    fn from(value: String) -> Self {
        Error::Generic(value)
    }
}

impl From<&str> for Error {
    fn from(value: &str) -> Self {
        Error::Generic(value.into())
    }
}
