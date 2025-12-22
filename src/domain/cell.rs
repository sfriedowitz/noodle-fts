use float_cmp::approx_eq;
use ndarray::{Array1, Array2, array};
use ndarray_linalg::{Determinant, Inverse};

use crate::utils::math::{HALF_PI, THIRD_PI};

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CellLattice {
    Lamellar,
    Square,
    Rectangular,
    Hexagonal2D,
    Oblique,
    Cubic,
    Tetragonal,
    Orthorhombic,
    Rhombohedral,
    Hexagonal3D,
    Monoclinic,
    Triclinic,
}

impl CellLattice {
    pub fn nparams(&self) -> usize {
        match self {
            Self::Lamellar => 1,
            Self::Square => 1,
            Self::Rectangular => 2,
            Self::Hexagonal2D => 1,
            Self::Oblique => 3,
            Self::Cubic => 1,
            Self::Tetragonal => 2,
            Self::Orthorhombic => 3,
            Self::Rhombohedral => 2,
            Self::Hexagonal3D => 2,
            Self::Monoclinic => 4,
            Self::Triclinic => 6,
        }
    }

    fn get_shape_tensor(&self, parameters: &Array1<f64>) -> Array2<f64> {
        match self {
            // 1D
            Self::Lamellar => shape_tensor_1d(parameters[0]),
            // 2D
            Self::Square => shape_tensor_2d(parameters[0], parameters[0], HALF_PI),
            Self::Rectangular => shape_tensor_2d(parameters[0], parameters[1], HALF_PI),
            Self::Hexagonal2D => shape_tensor_2d(parameters[0], parameters[0], THIRD_PI),
            Self::Oblique => shape_tensor_2d(parameters[0], parameters[1], parameters[2]),
            // 3D
            Self::Cubic => shape_tensor_3d(
                parameters[0],
                parameters[0],
                parameters[0],
                HALF_PI,
                HALF_PI,
                HALF_PI,
            ),
            Self::Tetragonal => shape_tensor_3d(
                parameters[0],
                parameters[0],
                parameters[1],
                HALF_PI,
                HALF_PI,
                HALF_PI,
            ),
            Self::Orthorhombic => shape_tensor_3d(
                parameters[0],
                parameters[1],
                parameters[2],
                HALF_PI,
                HALF_PI,
                HALF_PI,
            ),
            Self::Rhombohedral => shape_tensor_3d(
                parameters[0],
                parameters[0],
                parameters[0],
                parameters[1],
                parameters[1],
                parameters[1],
            ),
            Self::Hexagonal3D => shape_tensor_3d(
                parameters[0],
                parameters[0],
                parameters[1],
                HALF_PI,
                HALF_PI,
                THIRD_PI,
            ),
            Self::Monoclinic => shape_tensor_3d(
                parameters[0],
                parameters[1],
                parameters[2],
                HALF_PI,
                parameters[3],
                HALF_PI,
            ),
            Self::Triclinic => shape_tensor_3d(
                parameters[0],
                parameters[1],
                parameters[2],
                parameters[3],
                parameters[4],
                parameters[5],
            ),
        }
    }
}

#[derive(Debug, Clone)]
pub struct UnitCell {
    lattice: CellLattice,
    parameters: Array1<f64>,
}

impl UnitCell {
    // Constructors
    pub fn new(lattice: CellLattice, values: Array1<f64>) -> crate::Result<Self> {
        if values.len() != lattice.nparams() {
            return Err(crate::Error::Generic(format!(
                "Expected {} parameters for {:?}, got {}",
                lattice.nparams(),
                lattice,
                values.len()
            )));
        }
        Ok(Self {
            lattice,
            parameters: values,
        })
    }

    pub fn lamellar(a: f64) -> crate::Result<Self> {
        Self::new(CellLattice::Lamellar, array![a])
    }

    pub fn square(a: f64) -> crate::Result<Self> {
        Self::new(CellLattice::Square, array![a])
    }

    pub fn rectangular(a: f64, b: f64) -> crate::Result<Self> {
        Self::new(CellLattice::Rectangular, array![a, b])
    }

    pub fn hexagonal_2d(a: f64) -> crate::Result<Self> {
        Self::new(CellLattice::Hexagonal2D, array![a])
    }

    pub fn oblique(a: f64, b: f64, gamma: f64) -> crate::Result<Self> {
        Self::new(CellLattice::Oblique, array![a, b, gamma])
    }

    pub fn cubic(a: f64) -> crate::Result<Self> {
        Self::new(CellLattice::Cubic, array![a])
    }

    pub fn tetragonal(a: f64, c: f64) -> crate::Result<Self> {
        Self::new(CellLattice::Tetragonal, array![a, c])
    }

    pub fn orthorhombic(a: f64, b: f64, c: f64) -> crate::Result<Self> {
        Self::new(CellLattice::Orthorhombic, array![a, b, c])
    }

    pub fn rhombohedral(a: f64, alpha: f64) -> crate::Result<Self> {
        Self::new(CellLattice::Rhombohedral, array![a, alpha])
    }

    pub fn hexagonal_3d(a: f64, c: f64) -> crate::Result<Self> {
        Self::new(CellLattice::Hexagonal3D, array![a, c])
    }

    pub fn monoclinic(a: f64, b: f64, c: f64, beta: f64) -> crate::Result<Self> {
        Self::new(CellLattice::Monoclinic, array![a, b, c, beta])
    }

    pub fn triclinic(a: f64, b: f64, c: f64, alpha: f64, beta: f64, gamma: f64) -> crate::Result<Self> {
        Self::new(CellLattice::Triclinic, array![a, b, c, alpha, beta, gamma])
    }

    // Accessors
    pub fn lattice(&self) -> CellLattice {
        self.lattice
    }

    pub fn parameters(&self) -> &Array1<f64> {
        &self.parameters
    }

    pub fn nparams(&self) -> usize {
        self.parameters.len()
    }

    pub fn ndim(&self) -> usize {
        match self.lattice {
            CellLattice::Lamellar => 1,
            CellLattice::Square
            | CellLattice::Rectangular
            | CellLattice::Hexagonal2D
            | CellLattice::Oblique => 2,
            _ => 3,
        }
    }

    pub fn perturb(&self, idx: usize, delta: f64) -> crate::Result<Self> {
        if idx >= self.parameters.len() {
            return Err(crate::Error::Generic(format!(
                "Parameter index {} out of bounds (max {})",
                idx,
                self.parameters.len() - 1
            )));
        }
        let mut new_parameters = self.parameters.clone();
        new_parameters[idx] += delta;
        Ok(Self {
            lattice: self.lattice,
            parameters: new_parameters,
        })
    }

    pub fn shape(&self) -> Array2<f64> {
        self.lattice.get_shape_tensor(&self.parameters)
    }

    pub fn shape_inv(&self) -> Array2<f64> {
        self.shape().inv().expect("Shape tensor should be invertible")
    }

    pub fn metric(&self) -> Array2<f64> {
        let shape = self.shape();
        shape.t().dot(&shape)
    }

    pub fn metric_inv(&self) -> Array2<f64> {
        self.metric().inv().expect("Metric tensor should be invertible")
    }

    pub fn volume(&self) -> f64 {
        self.shape().det().expect("Shape tensor should be invertible")
    }
}

#[cfg(test)]
mod tests {

    use ndarray::Array2;

    use super::*;

    fn check_cell_inverses(cell: &UnitCell) {
        let eye = Array2::eye(cell.ndim());

        let cell_dot = cell.shape().dot(&cell.shape_inv());
        assert!(cell_dot.abs_diff_eq(&eye, 1e-8));

        let metric_dot = cell.metric().dot(&cell.metric_inv());
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

    #[test]
    fn test_volume_3d() {
        let cell = UnitCell::cubic(10.0).unwrap();
        assert!((cell.volume() - 1000.0).abs() < 1e-10);

        let cell = UnitCell::orthorhombic(10.0, 5.0, 2.0).unwrap();
        assert!((cell.volume() - 100.0).abs() < 1e-10);
    }
}
