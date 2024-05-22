use float_cmp::approx_eq;
use ndarray::{array, Array2};
use ndarray_linalg::Inverse;

use crate::{
    error::Result,
    math::{HALF_PI, THIRD_PI},
};

fn shape_tensor_1d(a: f64) -> Array2<f64> {
    array![[a]]
}

fn shape_tensor_2d(a: f64, b: f64, gamma: f64) -> Array2<f64> {
    let (cos_gamma, sin_gamma) = if approx_eq!(f64, gamma, HALF_PI) {
        (0.0, 1.0)
    } else {
        (gamma.cos(), gamma.sin())
    };

    let bx = b * cos_gamma;
    let by = b * sin_gamma;

    array![[a, bx], [0.0, by]]
}

fn shape_tensor_3d(a: f64, b: f64, c: f64, alpha: f64, beta: f64, gamma: f64) -> Array2<f64> {
    let cos_alpha = if approx_eq!(f64, alpha, HALF_PI) {
        0.0
    } else {
        alpha.cos()
    };
    let cos_beta = if approx_eq!(f64, beta, HALF_PI) {
        0.0
    } else {
        beta.cos()
    };
    let (cos_gamma, sin_gamma) = if approx_eq!(f64, gamma, HALF_PI) {
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

#[derive(Debug, Clone, Copy)]
pub enum CellParameters {
    // 1D
    Lamellar {
        a: f64,
    },
    // 2D
    Square {
        a: f64,
    },
    Rectangular {
        a: f64,
        b: f64,
    },
    Hexagonal2D {
        a: f64,
    },
    Oblique {
        a: f64,
        b: f64,
        gamma: f64,
    },
    // 3D
    Cubic {
        a: f64,
    },
    Tetragonal {
        a: f64,
        c: f64,
    },
    Orthorhombic {
        a: f64,
        b: f64,
        c: f64,
    },
    Rhombohedral {
        a: f64,
        alpha: f64,
    },
    Hexagonal3D {
        a: f64,
        c: f64,
    },
    Monoclinic {
        a: f64,
        b: f64,
        c: f64,
        beta: f64,
    },
    Triclinic {
        a: f64,
        b: f64,
        c: f64,
        alpha: f64,
        beta: f64,
        gamma: f64,
    },
}

impl CellParameters {
    pub fn into_shape_tensor(self) -> Array2<f64> {
        match self {
            // 1D
            Self::Lamellar { a } => shape_tensor_1d(a),
            // 2D
            Self::Square { a } => shape_tensor_2d(a, a, HALF_PI),
            Self::Rectangular { a, b } => shape_tensor_2d(a, b, HALF_PI),
            Self::Hexagonal2D { a } => shape_tensor_2d(a, a, THIRD_PI),
            Self::Oblique { a, b, gamma } => shape_tensor_2d(a, b, gamma),
            // 3D
            Self::Cubic { a } => shape_tensor_3d(a, a, a, HALF_PI, HALF_PI, HALF_PI),
            Self::Tetragonal { a, c } => shape_tensor_3d(a, a, c, HALF_PI, HALF_PI, HALF_PI),
            Self::Orthorhombic { a, b, c } => shape_tensor_3d(a, b, c, HALF_PI, HALF_PI, HALF_PI),
            Self::Rhombohedral { a, alpha } => shape_tensor_3d(a, a, a, alpha, alpha, alpha),
            Self::Hexagonal3D { a, c } => shape_tensor_3d(a, a, c, HALF_PI, HALF_PI, THIRD_PI),
            Self::Monoclinic { a, b, c, beta } => shape_tensor_3d(a, b, c, HALF_PI, beta, HALF_PI),
            Self::Triclinic {
                a,
                b,
                c,
                alpha,
                beta,
                gamma,
            } => shape_tensor_3d(a, b, c, alpha, beta, gamma),
        }
    }
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
    // Constructors
    pub fn new(parameters: CellParameters) -> Result<Self> {
        let shape = parameters.into_shape_tensor();
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

    pub fn lamellar(a: f64) -> Result<Self> {
        let parameters = CellParameters::Lamellar { a };
        Self::new(parameters)
    }

    pub fn square(a: f64) -> Result<Self> {
        let parameters = CellParameters::Square { a };
        Self::new(parameters)
    }

    pub fn rectangular(a: f64, b: f64) -> Result<Self> {
        let parameters = CellParameters::Rectangular { a, b };
        Self::new(parameters)
    }

    pub fn hexagonal_2d(a: f64) -> Result<Self> {
        let parameters = CellParameters::Hexagonal2D { a };
        Self::new(parameters)
    }

    pub fn oblique(a: f64, b: f64, gamma: f64) -> Result<Self> {
        let parameters = CellParameters::Oblique { a, b, gamma };
        Self::new(parameters)
    }

    pub fn cubic(a: f64) -> Result<Self> {
        let parameters = CellParameters::Cubic { a };
        Self::new(parameters)
    }

    pub fn tetragonal(a: f64, c: f64) -> Result<Self> {
        let parameters = CellParameters::Tetragonal { a, c };
        Self::new(parameters)
    }

    pub fn orthorhombic(a: f64, b: f64, c: f64) -> Result<Self> {
        let parameters = CellParameters::Orthorhombic { a, b, c };
        Self::new(parameters)
    }

    pub fn rhombohedral(a: f64, alpha: f64) -> Result<Self> {
        let parameters = CellParameters::Rhombohedral { a, alpha };
        Self::new(parameters)
    }

    pub fn hexagonal_3d(a: f64, c: f64) -> Result<Self> {
        let parameters = CellParameters::Hexagonal3D { a, c };
        Self::new(parameters)
    }

    pub fn monoclinic(a: f64, b: f64, c: f64, beta: f64) -> Result<Self> {
        let parameters = CellParameters::Monoclinic { a, b, c, beta };
        Self::new(parameters)
    }

    pub fn triclinic(a: f64, b: f64, c: f64, alpha: f64, beta: f64, gamma: f64) -> Result<Self> {
        let parameters = CellParameters::Triclinic {
            a,
            b,
            c,
            alpha,
            beta,
            gamma,
        };
        Self::new(parameters)
    }

    // Methods
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
}

#[cfg(test)]
mod tests {
    use ndarray::Array2;

    use super::{CellParameters, UnitCell};
    use crate::math::{HALF_PI, THIRD_PI};

    fn check_cell_inverses(cell: &UnitCell) {
        let eye = Array2::eye(cell.ndim());

        let cell_dot = cell.shape().dot(cell.shape_inv());
        assert!(cell_dot.abs_diff_eq(&eye, 1e-8));

        let metric_dot = cell.metric().dot(cell.metric_inv());
        assert!(metric_dot.abs_diff_eq(&eye, 1e-8));
    }

    fn check_cell_symmetry(cell: &UnitCell) {
        let metric = cell.metric();
        let metric_transpose = metric.t();
        assert!(metric.abs_diff_eq(&metric_transpose, 1e-8));

        let metric_inv = cell.metric_inv();
        let metric_inv_transpose = metric_inv.t();
        assert!(metric_inv.abs_diff_eq(&metric_inv_transpose, 1e-8));
    }

    #[test]
    fn test_1d_cells() {
        let cells = vec![UnitCell::lamellar(10.0).unwrap()];
        for cell in cells {
            assert!(cell.ndim() == 1);
            check_cell_inverses(&cell);
            check_cell_symmetry(&cell);
        }
    }

    #[test]
    fn test_2d_cells() {
        let cells = vec![
            UnitCell::square(10.0).unwrap(),
            UnitCell::rectangular(10.0, 5.0).unwrap(),
            UnitCell::hexagonal_2d(10.0).unwrap(),
            UnitCell::oblique(10.0, 5.0, THIRD_PI).unwrap(),
        ];
        for cell in cells {
            assert!(cell.ndim() == 2);
            check_cell_inverses(&cell);
            check_cell_symmetry(&cell);
        }
    }

    #[test]
    fn test_3d_cells() {
        let cells = vec![
            UnitCell::cubic(10.0).unwrap(),
            UnitCell::tetragonal(10.0, 5.0).unwrap(),
            UnitCell::orthorhombic(10.0, 7.0, 5.0).unwrap(),
            UnitCell::rhombohedral(10.0, HALF_PI).unwrap(),
            UnitCell::hexagonal_3d(10.0, 5.0).unwrap(),
            UnitCell::monoclinic(10.0, 10.0, 10.0, THIRD_PI).unwrap(),
            UnitCell::triclinic(10.0, 5.0, 2.0, HALF_PI, HALF_PI, THIRD_PI).unwrap(),
        ];
        for cell in cells {
            assert!(cell.ndim() == 3);
            check_cell_inverses(&cell);
            check_cell_symmetry(&cell);
        }
    }
}
