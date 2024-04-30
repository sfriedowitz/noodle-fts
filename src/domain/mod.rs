use ndarray::ArrayD;
use num::complex::Complex64;

pub mod cell;
pub mod domain;
pub mod fft;
pub mod mesh;

/// Real-valued multi-dimensional field grid.
pub type RField = ArrayD<f64>;

/// Complex-valued multi-dimensional field grid.
pub type CField = ArrayD<Complex64>;
