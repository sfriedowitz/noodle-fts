use std::ops::{Mul, MulAssign};

use ndarray::Zip;

use crate::fields::Field;

pub const PI: f64 = std::f64::consts::PI;
pub const HALF_PI: f64 = PI / 2.0;
pub const THIRD_PI: f64 = PI / 3.0;
pub const TWO_PI: f64 = 2.0 * PI;

pub fn apply_operator<A>(operator: &Field<A>, input: &Field<A>, output: &mut Field<A>)
where
    A: Clone + Mul<A, Output = A>,
{
    Zip::from(output)
        .and(input)
        .and(operator)
        .for_each(|output, input, op| *output = op.clone() * input.clone())
}

pub fn apply_operator_inplace<A, B>(operator: &Field<A>, output: &mut Field<B>)
where
    A: Clone,
    B: MulAssign<A>,
{
    Zip::from(output)
        .and(operator)
        .for_each(|output, op| *output *= op.clone())
}

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

#[cfg(test)]
mod tests {
    use ndarray::Array1;

    use super::{fftfreq, rfftfreq};

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
}
