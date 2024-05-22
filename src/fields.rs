use ndarray::{ArrayD, ArrayViewD};
use num::{complex::Complex64, Zero};

use crate::domain::Mesh;

/// Real-valued multi-dimensional field grid.
pub type RField = ArrayD<f64>;

/// Complex-valued multi-dimensional field grid.
pub type CField = ArrayD<Complex64>;
