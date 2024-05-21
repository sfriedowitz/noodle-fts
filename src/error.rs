#[derive(thiserror::Error, Debug)]
pub enum FTSError {
    #[error("Linalg: {0}")]
    Linalg(#[from] ndarray_linalg::error::LinalgError),
    #[error("Validation: {0}")]
    Validation(String),
    #[error("Generic: {0}")]
    Generic(String),
}
