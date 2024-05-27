use ndarray::ArrayD;
use num::complex::Complex64;

// Error handling
pub type Error = Box<dyn std::error::Error>;
pub type Result<T> = std::result::Result<T, Error>;

pub type Field<T> = ArrayD<T>;
// A multi-dimensional array with dynamic shape.

/// Real-valued multi-dimensional field grid.
pub type RField = Field<f64>;

/// Complex-valued multi-dimensional field grid.
pub type CField = Field<Complex64>;
