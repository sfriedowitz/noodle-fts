use thiserror::Error;

pub type Result<T> = ::std::result::Result<T, Error>;

#[derive(Error, Debug)]
pub enum Error {
    #[error("linear algebra error")]
    Linalg(#[from] ndarray_linalg::error::LinalgError),
    #[error("shape error")]
    ShapeMismatch(#[from] ndarray::ShapeError),
    #[error("dimension mismatch between '{0}' and '{1}'")]
    DimensionMismatch(String, String),
    #[error("validation failure: {0}")]
    Validation(String),
    #[error("generic error: {0}")]
    Generic(String),
}
