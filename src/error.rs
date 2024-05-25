use thiserror::Error;

pub type Result<T> = ::std::result::Result<T, FTSError>;

#[derive(Error, Debug)]
pub enum FTSError {
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
