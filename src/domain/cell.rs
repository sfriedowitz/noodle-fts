use float_cmp::approx_eq;
use ndarray::{array, Array2};
use ndarray_linalg::Inverse;

use crate::{
    math::{PI2, PI3},
    prelude::Result,
};

#[derive(Debug, Clone, Copy)]
pub enum CellParameters {
    // 1D
    Lamellar(f64),
    // 2D
    Square(f64),
    Rectangular(f64, f64),
    Hexagonal(f64),
    Oblique(f64, f64, f64),
    // 3D
    Cubic(f64),
    Tetragonal(f64, f64),
    Orthorhombic(f64, f64, f64),
    Trigonal(f64, f64),
    Hexagonal3D(f64, f64),
    Monoclinic(f64, f64, f64, f64),
    Triclinic(f64, f64, f64, f64, f64, f64),
}

#[derive(Debug, Clone)]
pub struct UnitCell {
    parameters: CellParameters,
    shape: Array2<f64>,
    shape_inv: Array2<f64>,
    metric: Array2<f64>,
    metric_inv: Array2<f64>,
}

impl UnitCell {
    pub fn new(parameters: CellParameters) -> Result<Self> {
        use CellParameters::*;

        let shape = match parameters {
            Lamellar(a) => Self::shape_1d(a),
            Square(a) => Self::shape_2d(a, a, PI2),
            Rectangular(a, b) => Self::shape_2d(a, b, PI2),
            Hexagonal(a) => Self::shape_2d(a, a, PI3),
            Oblique(a, b, gamma) => Self::shape_2d(a, b, gamma),
            Cubic(a) => Self::shape_3d(a, a, a, PI2, PI2, PI2),
            Tetragonal(a, c) => Self::shape_3d(a, a, c, PI2, PI2, PI2),
            Orthorhombic(a, b, c) => Self::shape_3d(a, b, c, PI2, PI2, PI2),
            Trigonal(a, alpha) => Self::shape_3d(a, a, a, alpha, alpha, alpha),
            Hexagonal3D(a, c) => Self::shape_3d(a, a, c, PI2, PI2, PI3),
            Monoclinic(a, b, c, beta) => Self::shape_3d(a, b, c, PI2, beta, PI2),
            Triclinic(a, b, c, alpha, beta, gamma) => Self::shape_3d(a, b, c, alpha, beta, gamma),
        };

        let shape_inv = shape.inv()?;
        let metric = shape.dot(&shape.t());
        let metric_inv = metric.inv()?;

        Ok(Self {
            parameters,
            shape,
            shape_inv,
            metric,
            metric_inv,
        })
    }

    pub fn ndim(&self) -> usize {
        self.shape.shape()[0]
    }

    pub fn parameters(&self) -> CellParameters {
        self.parameters
    }

    pub fn shape(&self) -> &Array2<f64> {
        &self.shape
    }

    pub fn shape_inv(&self) -> &Array2<f64> {
        &self.shape_inv
    }

    pub fn metric(&self) -> &Array2<f64> {
        &self.metric
    }

    pub fn metric_inv(&self) -> &Array2<f64> {
        &self.metric_inv
    }

    fn shape_1d(a: f64) -> Array2<f64> {
        array![[a]]
    }

    fn shape_2d(a: f64, b: f64, gamma: f64) -> Array2<f64> {
        let (cos_gamma, sin_gamma) = if approx_eq!(f64, gamma, PI2) {
            (0.0, 1.0)
        } else {
            (gamma.cos(), gamma.sin())
        };

        let bx = b * cos_gamma;
        let by = b * sin_gamma;

        array![[a, bx], [0.0, by]]
    }

    fn shape_3d(a: f64, b: f64, c: f64, alpha: f64, beta: f64, gamma: f64) -> Array2<f64> {
        let cos_alpha = if approx_eq!(f64, alpha, PI2) {
            0.0
        } else {
            alpha.cos()
        };
        let cos_beta = if approx_eq!(f64, beta, PI2) {
            0.0
        } else {
            beta.cos()
        };
        let (cos_gamma, sin_gamma) = if approx_eq!(f64, gamma, PI2) {
            (0.0, 1.0)
        } else {
            (gamma.cos(), gamma.sin())
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
    use crate::math::PI2;

    fn check_cell_shape(cell: &UnitCell) {
        let eye = Array2::eye(cell.ndim());

        let cell_dot = cell.shape().dot(cell.shape_inv());
        for (x, i) in cell_dot.iter().zip(eye.iter()) {
            assert_approx_eq!(f64, *x, *i)
        }

        let metric_dot = cell.metric().dot(cell.metric_inv());
        for (x, i) in metric_dot.iter().zip(eye.iter()) {
            assert_approx_eq!(f64, *x, *i)
        }
    }

    #[test]
    fn test_1d() {
        let parameters = CellParameters::Lamellar(10.0);
        let cell = UnitCell::new(parameters).unwrap();
        assert!(cell.ndim() == 1);
        check_cell_shape(&cell);
    }

    #[test]
    fn test_2d() {
        let parameters = CellParameters::Oblique(10.0, 5.0, PI2 - 0.1);
        let cell = UnitCell::new(parameters).unwrap();
        assert!(cell.ndim() == 2);
        check_cell_shape(&cell);
    }

    #[test]
    fn test_3d() {
        let parameters = CellParameters::Triclinic(10.0, 5.0, 2.0, PI2 - 0.1, PI2 + 0.1, PI2);
        let cell = UnitCell::new(parameters).unwrap();
        assert!(cell.ndim() == 3);
        check_cell_shape(&cell);

        dbg!(cell);
    }
}
