use float_cmp::approx_eq;
use ndarray::{array, Array2};
use ndarray_linalg::Inverse;

use crate::math::PI2;

#[derive(Debug, Clone, Copy)]
pub enum CellParameters {
    OneD(f64),
    TwoD {
        a: f64,
        b: f64,
        gamma: f64,
    },
    ThreeD {
        a: f64,
        b: f64,
        c: f64,
        alpha: f64,
        beta: f64,
        gamma: f64,
    },
}

#[derive(Debug, Clone)]
pub struct UnitCell {
    parameters: CellParameters,
    cell: Array2<f64>,
    cell_inv: Array2<f64>,
    metric: Array2<f64>,
    metric_inv: Array2<f64>,
}

impl UnitCell {
    pub fn new(parameters: CellParameters) -> Self {
        let cell = match parameters {
            CellParameters::OneD(length) => UnitCell::make_1d(length),
            CellParameters::TwoD { a, b, gamma } => UnitCell::make_2d(a, b, gamma),
            CellParameters::ThreeD {
                a,
                b,
                c,
                alpha,
                beta,
                gamma,
            } => UnitCell::make_3d(a, b, c, alpha, beta, gamma),
        };
        let cell_inv = cell.inv().unwrap();
        let metric = cell.dot(&cell.t());
        let metric_inv = cell_inv.dot(&cell_inv.t());
        Self {
            parameters,
            cell,
            cell_inv,
            metric,
            metric_inv,
        }
    }

    pub fn n_dim(&self) -> usize {
        self.cell.shape()[0]
    }

    fn make_1d(a: f64) -> Array2<f64> {
        array![[a]]
    }

    fn make_2d(a: f64, b: f64, gamma: f64) -> Array2<f64> {
        let (cos_gamma, sin_gamma) = match approx_eq!(f64, gamma, PI2) {
            true => (0.0, 1.0),
            false => (gamma.cos(), gamma.sin()),
        };
        let bx = b * cos_gamma;
        let by = b * sin_gamma;
        array![[a, bx], [0.0, by]]
    }

    fn make_3d(a: f64, b: f64, c: f64, alpha: f64, beta: f64, gamma: f64) -> Array2<f64> {
        let cos_alpha = match approx_eq!(f64, alpha, PI2) {
            true => 0.0,
            false => alpha.cos(),
        };
        let cos_beta = match approx_eq!(f64, beta, PI2) {
            true => 0.0,
            false => beta.cos(),
        };
        let (cos_gamma, sin_gamma) = match approx_eq!(f64, gamma, PI2) {
            true => (0.0, 1.0),
            false => (gamma.cos(), gamma.sin()),
        };

        let ax = a;
        let bx = b * cos_gamma;
        let by = b * sin_gamma;
        let cx = c * cos_beta;
        let cy = c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma;
        let cz = (c * c - cx * cx - cy * cy).sqrt();

        array![[ax, bx, cx], [0.0, by, cy], [0.0, 0.0, cz]]
    }
}

#[cfg(test)]
mod tests {
    use float_cmp::assert_approx_eq;
    use ndarray::Array2;

    use super::{CellParameters, UnitCell};
    use crate::domain::cell::PI2;

    fn check_cell_inverses(cell: &UnitCell) {
        let eye = Array2::eye(cell.n_dim());

        let cell_dot = cell.cell.dot(&cell.cell_inv);
        for (x, i) in cell_dot.iter().zip(eye.iter()) {
            assert_approx_eq!(f64, *x, *i)
        }

        let metric_dot = cell.metric.dot(&cell.metric_inv);
        for (x, i) in metric_dot.iter().zip(eye.iter()) {
            assert_approx_eq!(f64, *x, *i)
        }
    }

    #[test]
    fn test_1d() {
        let parameters = CellParameters::OneD(10.0);
        let cell = UnitCell::new(parameters);
        assert!(cell.n_dim() == 1);
        check_cell_inverses(&cell);
    }

    #[test]
    fn test_2d() {
        let parameters = CellParameters::TwoD {
            a: 10.0,
            b: 5.0,
            gamma: PI2,
        };
        let cell = UnitCell::new(parameters);
        assert!(cell.n_dim() == 2);
        check_cell_inverses(&cell);
    }

    #[test]
    fn test_3d() {
        let parameters = CellParameters::ThreeD {
            a: 10.0,
            b: 5.0,
            c: 2.0,
            alpha: PI2,
            beta: PI2,
            gamma: PI2,
        };
        let cell = UnitCell::new(parameters);
        assert!(cell.n_dim() == 3);
        check_cell_inverses(&cell);
    }
}
