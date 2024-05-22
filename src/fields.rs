use ndarray::{ArrayD, ArrayViewD};
use num::{complex::Complex64, Zero};

use crate::domain::Mesh;

pub type Field<T> = ArrayD<T>;
// A multi-dimensional array with dynamic shape.

/// Real-valued multi-dimensional field grid.
pub type RField = Field<f64>;

/// Complex-valued multi-dimensional field grid.
pub type CField = Field<Complex64>;
