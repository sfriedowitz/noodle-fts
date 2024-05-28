use ndarray::ArrayD;
use num::complex::Complex64;

/// A multi-dimensional array with dynamic shape.
pub type Field<T> = ArrayD<T>;

/// Real-valued multi-dimensional field grid.
pub type RField = Field<f64>;

/// Complex-valued multi-dimensional field grid.
pub type CField = Field<Complex64>;
