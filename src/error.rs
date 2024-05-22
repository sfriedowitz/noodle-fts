#[derive(thiserror::Error, Debug)]
pub enum FTSError {
    #[error("linear algebra error")]
    Linalg(#[from] ndarray_linalg::error::LinalgError),
    #[error("number of dimensions do not match: {0} != {1}")]
    DimensionMismatch(usize, usize),
    #[error("validation failure -- {0}")]
    Validation(String),
    #[error("generic error: {0}")]
    Generic(String),
}

pub type Result<T> = core::result::Result<T, FTSError>;
