use ndarray::{Array1, Axis};

use crate::RField;

pub const PI: f64 = std::f64::consts::PI;
pub const HALF_PI: f64 = PI / 2.0;
pub const THIRD_PI: f64 = PI / 3.0;
pub const TWO_PI: f64 = 2.0 * PI;

pub fn fftshift_index(idx: usize, n: usize) -> f64 {
    if idx <= n / 2 - 1 {
        idx as f64
    } else {
        idx as f64 - n as f64
    }
}

pub fn fftfreq(n: usize, d: Option<f64>) -> impl Iterator<Item = f64> + Clone {
    let d = d.unwrap_or(1.0);
    let norm = d * n as f64;
    (0..n).map(move |i| fftshift_index(i, n) / norm)
}

pub fn rfftfreq(n: usize, d: Option<f64>) -> impl Iterator<Item = f64> + Clone {
    let d = d.unwrap_or(1.0);
    let norm = d * n as f64;
    (0..n / 2 + 1).map(move |i| i as f64 / norm)
}

pub fn simpsons(x: &RField, dx: Option<f64>, axis: Axis) -> RField {
    // This probably is inaccurate if the broadcast dimension is ambiguous
    let dx = dx.unwrap_or(1.0);

    let n = x.shape()[axis.0];
    let coef = Array1::from_iter((0..n).map(|i| {
        if i == 0 || i == n - 1 {
            1.0
        } else if i % 2 == 0 {
            2.0
        } else {
            4.0
        }
    }));

    (dx / 3.0) * (coef * x).sum_axis(axis)
}

#[cfg(test)]
mod tests {

    use ndarray::{Array1, Axis};

    use super::*;

    #[test]
    fn test_fftfreq() {
        let got = Array1::from_iter(fftfreq(4, Some(0.25)));
        let expected = Array1::from_vec(vec![0.0, 1.0, -2.0, -1.0]);
        assert!(got.abs_diff_eq(&expected, 1e-8));
    }

    #[test]
    fn test_rfftfreq() {
        let got = Array1::from_iter(rfftfreq(4, Some(0.25)));
        let expected = Array1::from_vec(vec![0.0, 1.0, 2.0]);
        assert!(got.abs_diff_eq(&expected, 1e-8));
    }

    #[test]
    fn test_simpsons_integration() {
        let x = Array1::linspace(0.0, 1.0, 11).into_dyn();
        let x2 = &x * &x;
        let output = simpsons(&x2, Some(0.1), Axis(0));
        dbg!(x);
        dbg!(output);
    }
}
