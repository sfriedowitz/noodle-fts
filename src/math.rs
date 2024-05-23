use ndarray::Zip;

use crate::fields::{Field, RField};

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

pub fn simpsons_product(xfields: &[RField], yfields: &[RField], dx: Option<f64>) -> RField {
    let nf = xfields.len();
    assert!(yfields.len() == nf);
    let dx = dx.unwrap_or(1.0);

    // Reset
    let mut output = RField::zeros(xfields[0].shape());

    // Endpoint contributions
    Zip::from(&mut output)
        .and(&xfields[0])
        .and(&yfields[nf - 1])
        .for_each(|out, x, y| *out += x * y);
    Zip::from(&mut output)
        .and(&xfields[nf - 1])
        .and(&yfields[0])
        .for_each(|out, x, y| *out += x * y);

    // Odd indices
    for i in (1..nf - 1).step_by(2) {
        Zip::from(&mut output)
            .and(&xfields[i])
            .and(&yfields[nf - i - 1])
            .for_each(|out, x, y| *out += x * y);
    }

    // Even indices
    for i in (2..nf - 2).step_by(2) {
        Zip::from(&mut output)
            .and(&xfields[i])
            .and(&yfields[nf - i - 1])
            .for_each(|out, x, y| *out += x * y);
    }

    // Normalize the integral
    output *= dx / 3.0;

    output
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
