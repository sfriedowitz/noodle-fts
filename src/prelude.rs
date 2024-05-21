use ndarray::ArrayD;
use num::complex::Complex64;

pub use crate::error::FTSError;

pub type Result<T> = core::result::Result<T, FTSError>;

/// Real-valued multi-dimensional field grid.
pub type RField = ArrayD<f64>;

/// Complex-valued multi-dimensional field grid.
pub type CField = ArrayD<Complex64>;
