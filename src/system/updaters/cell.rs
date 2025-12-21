use ndarray::Array2;

use crate::{
    Error, Result,
    domain::{CellParameters, UnitCell},
    system::System,
};

/// Cell parameter updater using stress-based gradient descent.
///
/// This updater implements a damped gradient descent algorithm (Parrinello-Rahman style)
/// to optimize unit cell parameters based on the stress tensor. The equilibrium condition
/// is a stress-free crystal (stress tensor = 0).
///
/// The update rule is: `params_new = params_old - damping * stress`
///
/// # Algorithm
/// - First-order damped dynamics
/// - No predictor-corrector (parameters are scalars, not fields)
/// - Intelligent stress-to-parameter mapping based on cell type
///
/// # References
/// - Parrinello-Rahman cell relaxation dynamics
/// - Fredrickson's "The Equilibrium Theory of Inhomogeneous Polymers"
#[derive(Debug, Clone)]
pub struct CellUpdater {
    damping: f64,
}

impl CellUpdater {
    /// Create a new cell updater with the specified damping coefficient.
    ///
    /// # Parameters
    /// - `damping`: Damping coefficient (typical range: 0.2-2.0)
    ///   - Larger values: faster convergence but risk of oscillations
    ///   - Smaller values: slower but more stable
    ///   - Default recommendation: 1.0
    pub fn new(damping: f64) -> Self {
        Self { damping }
    }

    /// Get the current damping coefficient.
    pub fn damping(&self) -> f64 {
        self.damping
    }

    /// Update shape tensor using stress and extract new parameters.
    ///
    /// The stress tensor σ and shape tensor h have the same dimensions.
    /// We update the shape tensor directly: h_new = h_old - damping * σ
    /// Then extract parameters from the updated shape tensor.
    fn update_parameters(
        &self,
        params: CellParameters,
        stress: &[f64],
        shape: &Array2<f64>,
    ) -> Result<CellParameters> {
        let ndim = shape.shape()[0];

        // Convert flattened stress vector to symmetric matrix
        let stress_matrix = self.stress_to_matrix(stress, ndim);

        // Update shape tensor: h_new = h - damping * σ
        let mut shape_new = shape.clone();
        for i in 0..ndim {
            for j in 0..ndim {
                shape_new[[i, j]] -= self.damping * stress_matrix[[i, j]];
            }
        }

        // Extract parameters from updated shape tensor based on cell type
        self.extract_parameters(params, &shape_new)
    }

    /// Convert flattened stress vector to symmetric matrix.
    ///
    /// Stress is stored as [σ_xx, σ_yy, σ_zz, σ_xy, σ_xz, σ_yz] (only unique components).
    /// We reconstruct the full symmetric matrix.
    fn stress_to_matrix(&self, stress: &[f64], ndim: usize) -> Array2<f64> {
        let mut matrix = Array2::zeros((ndim, ndim));

        match ndim {
            1 => {
                matrix[[0, 0]] = stress[0];
            }
            2 => {
                matrix[[0, 0]] = stress[0]; // σ_xx
                matrix[[1, 1]] = stress[1]; // σ_yy
                matrix[[0, 1]] = stress[2]; // σ_xy
                matrix[[1, 0]] = stress[2]; // σ_xy (symmetric)
            }
            3 => {
                matrix[[0, 0]] = stress[0]; // σ_xx
                matrix[[1, 1]] = stress[1]; // σ_yy
                matrix[[2, 2]] = stress[2]; // σ_zz
                matrix[[0, 1]] = stress[3]; // σ_xy
                matrix[[1, 0]] = stress[3]; // σ_xy
                matrix[[0, 2]] = stress[4]; // σ_xz
                matrix[[2, 0]] = stress[4]; // σ_xz
                matrix[[1, 2]] = stress[5]; // σ_yz
                matrix[[2, 1]] = stress[5]; // σ_yz
            }
            _ => {}
        }

        matrix
    }

    /// Extract cell parameters from updated shape tensor.
    ///
    /// Different cell types extract different parameters from the shape tensor.
    /// Constrained cell types (Square, Cubic) average to maintain constraints.
    fn extract_parameters(&self, params: CellParameters, shape: &Array2<f64>) -> Result<CellParameters> {
        match params {
            // 1D cells
            CellParameters::Lamellar { .. } => Ok(CellParameters::Lamellar {
                a: shape[[0, 0]].max(0.1),
            }),

            // 2D cells
            CellParameters::Square { .. } => {
                // Average diagonal to maintain a = b
                let avg = (shape[[0, 0]] + shape[[1, 1]]) / 2.0;
                Ok(CellParameters::Square { a: avg.max(0.1) })
            }
            CellParameters::Rectangular { .. } => Ok(CellParameters::Rectangular {
                a: shape[[0, 0]].max(0.1),
                b: shape[[1, 1]].max(0.1),
            }),
            CellParameters::Hexagonal2D { .. } => {
                // Average diagonal to maintain constraint
                let avg = (shape[[0, 0]] + shape[[1, 1]]) / 2.0;
                Ok(CellParameters::Hexagonal2D { a: avg.max(0.1) })
            }
            CellParameters::Oblique { .. } => {
                let a = shape[[0, 0]];
                let b_x = shape[[0, 1]];
                let b_y = shape[[1, 1]];
                let b = (b_x * b_x + b_y * b_y).sqrt();
                let gamma = (b_x / b).acos();
                Ok(CellParameters::Oblique {
                    a: a.max(0.1),
                    b: b.max(0.1),
                    gamma: gamma.clamp(0.1, std::f64::consts::PI - 0.1),
                })
            }

            // 3D cells
            CellParameters::Cubic { .. } => {
                // Average diagonal to maintain a = b = c
                let avg = (shape[[0, 0]] + shape[[1, 1]] + shape[[2, 2]]) / 3.0;
                Ok(CellParameters::Cubic { a: avg.max(0.1) })
            }
            CellParameters::Tetragonal { .. } => {
                let avg_ab = (shape[[0, 0]] + shape[[1, 1]]) / 2.0;
                Ok(CellParameters::Tetragonal {
                    a: avg_ab.max(0.1),
                    c: shape[[2, 2]].max(0.1),
                })
            }
            CellParameters::Orthorhombic { .. } => Ok(CellParameters::Orthorhombic {
                a: shape[[0, 0]].max(0.1),
                b: shape[[1, 1]].max(0.1),
                c: shape[[2, 2]].max(0.1),
            }),

            // Complex 3D cells not yet supported
            _ => Err(Error::Generic(Box::from(
                "Cell type not yet supported for variable cell optimization",
            ))),
        }
    }
}

impl super::SystemUpdater for CellUpdater {
    /// Perform one cell parameter update step.
    ///
    /// This method:
    /// 1. Updates cell parameters based on stress: p_new = p_old - damping * σ
    /// 2. Reconstructs the UnitCell with new parameters
    /// 3. Updates the system state with new cell
    ///
    /// # Returns
    /// - `Ok(())` on success
    /// - `Err(_)` if cell type is unsupported or parameters are invalid
    fn step(&mut self, system: &mut System) -> Result<()> {
        let stress = system.stress();

        // Get current cell parameters and shape tensor
        let cell = system.domain().cell();
        let params = cell.parameters();
        let shape = cell.shape();

        // Update parameters based on stress and shape tensor
        let new_params = self.update_parameters(params, stress, shape)?;

        // Reconstruct cell with new parameters
        let new_cell = UnitCell::new(new_params)?;

        // Assign new cell to system
        *system.domain_mut().cell_mut() = new_cell;

        // Recompute system state with new cell (updates ksq, concentrations, etc.)
        system.update();

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use float_cmp::assert_approx_eq;

    use super::*;
    use crate::{
        chem::{Block, Monomer, Polymer},
        domain::Mesh,
        system::SystemUpdater,
    };

    fn create_test_system() -> System {
        let mesh = Mesh::One(32);
        let cell = UnitCell::lamellar(10.0).unwrap();

        let monomer_a = Monomer::new(0, 1.0);
        let monomer_b = Monomer::new(1, 1.0);

        let block_a = Block::new(monomer_a, 50, 1.0);
        let block_b = Block::new(monomer_b, 50, 1.0);

        let polymer = Polymer::new(vec![block_a, block_b], 100, 1.0);

        System::new(mesh, cell, vec![polymer.into()]).unwrap()
    }

    #[test]
    fn test_cell_updater_step() {
        let mut system = create_test_system();
        system.update();

        let mut updater = CellUpdater::new(0.1);

        // Should be able to call step without error
        let result = updater.step(&mut system);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cell_updater_maintains_positive_params() {
        let mut system = create_test_system();
        system.update();

        let mut updater = CellUpdater::new(1.0);

        // Run several steps
        for _ in 0..10 {
            updater.step(&mut system).unwrap();
        }

        // Check that cell parameter is still positive
        let cell = system.domain().cell();
        match cell.parameters() {
            CellParameters::Lamellar { a } => {
                assert!(a > 0.0, "Cell parameter should remain positive");
            }
            _ => panic!("Expected Lamellar cell"),
        }
    }

    #[test]
    fn test_update_parameters_lamellar() {
        let updater = CellUpdater::new(1.0);
        let cell = UnitCell::lamellar(10.0).unwrap();
        let params = cell.parameters();
        let shape = cell.shape();
        let stress = vec![2.0]; // Positive stress → decrease a

        let new_params = updater.update_parameters(params, &stress, shape).unwrap();

        match new_params {
            CellParameters::Lamellar { a } => {
                assert!(a < 10.0, "Parameter should decrease with positive stress");
                assert!(a > 0.0, "Parameter should remain positive");
            }
            _ => panic!("Expected Lamellar parameters"),
        }
    }

    #[test]
    fn test_update_parameters_square() {
        let updater = CellUpdater::new(1.0);
        let cell = UnitCell::square(10.0).unwrap();
        let params = cell.parameters();
        let shape = cell.shape();
        let stress = vec![1.0, 1.0, 0.0]; // Equal diagonal stress

        let new_params = updater.update_parameters(params, &stress, shape).unwrap();

        match new_params {
            CellParameters::Square { a } => {
                // Average stress = 1.0, so a should decrease by 1.0
                assert_approx_eq!(f64, a, 9.0, epsilon = 1e-10);
            }
            _ => panic!("Expected Square parameters"),
        }
    }

    #[test]
    fn test_update_parameters_rectangular() {
        let updater = CellUpdater::new(1.0);
        let cell = UnitCell::rectangular(10.0, 8.0).unwrap();
        let params = cell.parameters();
        let shape = cell.shape();
        let stress = vec![2.0, 1.0, 0.0]; // Different diagonal stresses

        let new_params = updater.update_parameters(params, &stress, shape).unwrap();

        match new_params {
            CellParameters::Rectangular { a, b } => {
                assert_approx_eq!(f64, a, 8.0, epsilon = 1e-10); // 10 - 2
                assert_approx_eq!(f64, b, 7.0, epsilon = 1e-10); // 8 - 1
            }
            _ => panic!("Expected Rectangular parameters"),
        }
    }

    #[test]
    fn test_update_parameters_cubic() {
        let updater = CellUpdater::new(1.0);
        let cell = UnitCell::cubic(10.0).unwrap();
        let params = cell.parameters();
        let shape = cell.shape();
        let stress = vec![1.0, 2.0, 3.0, 0.0, 0.0, 0.0]; // Average = 2.0

        let new_params = updater.update_parameters(params, &stress, shape).unwrap();

        match new_params {
            CellParameters::Cubic { a } => {
                assert_approx_eq!(f64, a, 8.0, epsilon = 1e-10); // 10 - 2
            }
            _ => panic!("Expected Cubic parameters"),
        }
    }

    #[test]
    fn test_update_parameters_orthorhombic() {
        let updater = CellUpdater::new(1.0);
        let cell = UnitCell::orthorhombic(10.0, 8.0, 6.0).unwrap();
        let params = cell.parameters();
        let shape = cell.shape();
        let stress = vec![1.0, 2.0, 3.0, 0.0, 0.0, 0.0];

        let new_params = updater.update_parameters(params, &stress, shape).unwrap();

        match new_params {
            CellParameters::Orthorhombic { a, b, c } => {
                assert_approx_eq!(f64, a, 9.0, epsilon = 1e-10); // 10 - 1
                assert_approx_eq!(f64, b, 6.0, epsilon = 1e-10); // 8 - 2
                assert_approx_eq!(f64, c, 3.0, epsilon = 1e-10); // 6 - 3
            }
            _ => panic!("Expected Orthorhombic parameters"),
        }
    }

    #[test]
    fn test_hexagonal_cell_supported() {
        let updater = CellUpdater::new(1.0);
        let cell = UnitCell::hexagonal_2d(10.0).unwrap();
        let params = cell.parameters();
        let shape = cell.shape();
        let stress = vec![1.0, 1.0, 0.0];

        let result = updater.update_parameters(params, &stress, shape);
        assert!(result.is_ok(), "Hexagonal2D should now be supported");

        match result.unwrap() {
            CellParameters::Hexagonal2D { a } => {
                // Should average diagonal stress
                assert!(a < 10.0);
            }
            _ => panic!("Expected Hexagonal2D parameters"),
        }
    }
}
