use ndarray::{Array1, Array2, Array3};

use crate::{
    domain::{CellLattice, UnitCell},
    system::System,
};

/// Computes Jacobian for cell parameter updates using finite differences.
///
/// The Jacobian relates changes in the shape tensor to changes in cell parameters:
/// J[i,j,k] = ∂h[i,j]/∂param[k]
///
/// This is computed using forward finite differences:
/// J[i,j,k] ≈ (h(params + δe_k) - h(params)) / δ
///
/// The Jacobian is then used to project stress gradients to parameter space via:
/// grad_params[k] = Σ_{i,j} J[i,j,k] * stress[i,j]
#[derive(Debug, Clone)]
pub struct CellJacobian {
    epsilon: f64,
}

impl CellJacobian {
    /// Create a new Jacobian computer with the specified finite difference step.
    ///
    /// # Parameters
    /// - `epsilon`: Finite difference step size (typical: 1e-6)
    pub fn new(epsilon: f64) -> Self {
        Self { epsilon }
    }

    /// Compute the Jacobian ∂h/∂params using finite differences.
    ///
    /// Returns Array3<f64> of shape (ndim, ndim, nparams) where:
    /// - result[[i, j, k]] = ∂h[i,j]/∂param[k]
    ///
    /// # Parameters
    /// - `cell`: The unit cell at which to compute the Jacobian
    pub fn compute(&self, cell: &UnitCell) -> crate::Result<Array3<f64>> {
        let ndim = cell.ndim();
        let nparams = cell.nparams();
        let h0 = cell.shape();

        let mut jacobian = Array3::zeros((ndim, ndim, nparams));

        for k in 0..nparams {
            // Perturb parameter k
            let cell_perturbed = cell.perturb(k, self.epsilon)?;
            let h_perturbed = cell_perturbed.shape();

            // Compute finite difference: (h_perturbed - h0) / epsilon
            for i in 0..ndim {
                for j in 0..ndim {
                    jacobian[[i, j, k]] = (h_perturbed[[i, j]] - h0[[i, j]]) / self.epsilon;
                }
            }
        }

        Ok(jacobian)
    }

    /// Project stress tensor to parameter gradient via tensor contraction.
    ///
    /// Computes: grad_params[k] = Σ_{i,j} J[i,j,k] * stress[i,j]
    ///
    /// # Parameters
    /// - `jacobian`: The Jacobian from compute()
    /// - `stress`: The stress tensor (ndim × ndim)
    ///
    /// # Returns
    /// Array1<f64> of length nparams containing the gradient in parameter space
    pub fn project_stress(&self, jacobian: &Array3<f64>, stress: &Array2<f64>) -> Array1<f64> {
        let shape = jacobian.shape();
        let ndim = shape[0];
        let nparams = shape[2];

        let mut grad_params = Array1::zeros(nparams);

        for k in 0..nparams {
            let mut sum = 0.0;
            for i in 0..ndim {
                for j in 0..ndim {
                    sum += jacobian[[i, j, k]] * stress[[i, j]];
                }
            }
            grad_params[k] = sum;
        }

        grad_params
    }
}

/// Cell parameter updater using stress-based gradient descent.
///
/// This updater implements a damped gradient descent algorithm (Parrinello-Rahman style)
/// to optimize unit cell parameters based on the stress tensor. The equilibrium condition
/// is a stress-free crystal (stress tensor = 0).
///
/// The update rule uses finite difference Jacobian projection:
/// 1. Compute J = ∂h/∂params using finite differences
/// 2. Project stress: grad = J^T · σ
/// 3. Update: params_new = params_old - step_size * grad
#[derive(Debug, Clone)]
pub struct CellUpdater {
    step_size: f64,
    jacobian: CellJacobian,
}

impl CellUpdater {
    pub fn new(step_size: f64) -> Self {
        Self {
            step_size,
            jacobian: CellJacobian::new(1e-6),
        }
    }

    pub fn step_size(&self) -> f64 {
        self.step_size
    }

    fn update_cell(&self, cell: &UnitCell, stress: &Array2<f64>) -> crate::Result<UnitCell> {
        // 1. Compute Jacobian
        let jacobian = self.jacobian.compute(cell)?;

        // 2. Project stress to parameter gradient
        let grad_params = self.jacobian.project_stress(&jacobian, stress);

        // 3. Update parameters (vectorized)
        let mut new_parameters = cell.parameters() - self.step_size * &grad_params;

        // 4. Enforce bounds
        Self::clamp_parameters(&mut new_parameters, cell.lattice());

        // 5. Construct new cell
        UnitCell::new(cell.lattice(), new_parameters)
    }

    /// Clamp parameters to physically valid ranges.
    ///
    /// - Lengths must be > 0.1 (prevents collapse)
    /// - Angles must be in (0.1, π - 0.1) (prevents degenerate cells)
    fn clamp_parameters(parameters: &mut Array1<f64>, lattice: CellLattice) {
        match lattice {
            // Length-only cells
            CellLattice::Lamellar | CellLattice::Square | CellLattice::Hexagonal2D | CellLattice::Cubic => {
                for p in parameters.iter_mut() {
                    *p = p.max(0.1);
                }
            }
            // Length-only cells (multiple lengths)
            CellLattice::Rectangular | CellLattice::Tetragonal | CellLattice::Orthorhombic => {
                for p in parameters.iter_mut() {
                    *p = p.max(0.1);
                }
            }
            // Oblique: [a, b, gamma]
            CellLattice::Oblique => {
                parameters[0] = parameters[0].max(0.1);
                parameters[1] = parameters[1].max(0.1);
                parameters[2] = parameters[2].clamp(0.1, std::f64::consts::PI - 0.1);
            }
            // Rhombohedral: [a, alpha]
            CellLattice::Rhombohedral => {
                parameters[0] = parameters[0].max(0.1);
                parameters[1] = parameters[1].clamp(0.1, std::f64::consts::PI - 0.1);
            }
            // Monoclinic: [a, b, c, beta]
            CellLattice::Monoclinic => {
                parameters[0] = parameters[0].max(0.1);
                parameters[1] = parameters[1].max(0.1);
                parameters[2] = parameters[2].max(0.1);
                parameters[3] = parameters[3].clamp(0.1, std::f64::consts::PI - 0.1);
            }
            // Triclinic: [a, b, c, alpha, beta, gamma]
            CellLattice::Triclinic => {
                parameters[0] = parameters[0].max(0.1);
                parameters[1] = parameters[1].max(0.1);
                parameters[2] = parameters[2].max(0.1);
                parameters[3] = parameters[3].clamp(0.1, std::f64::consts::PI - 0.1);
                parameters[4] = parameters[4].clamp(0.1, std::f64::consts::PI - 0.1);
                parameters[5] = parameters[5].clamp(0.1, std::f64::consts::PI - 0.1);
            }
            // Hexagonal3D: [a, c]
            CellLattice::Hexagonal3D => {
                parameters[0] = parameters[0].max(0.1);
                parameters[1] = parameters[1].max(0.1);
            }
        }
    }
}

impl super::SystemUpdater for CellUpdater {
    fn step(&mut self, system: &mut System) -> crate::Result<()> {
        let stress = system.stress();
        let cell = system.domain().cell();

        // Update cell based on stress
        let new_cell = self.update_cell(cell, stress)?;

        // Assign new cell to system
        system.domain_mut().set_cell(new_cell);

        // Recompute system state with new cell (updates ksq, concentrations, etc.)
        system.update();

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use float_cmp::assert_approx_eq;
    use ndarray::array;

    use super::*;
    use crate::{
        chem::{Block, Monomer, Polymer},
        domain::{CellLattice, Mesh},
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
    fn test_jacobian_lamellar() {
        let jacobian = CellJacobian::new(1e-6);
        let cell = UnitCell::lamellar(10.0).unwrap();

        let j = jacobian.compute(&cell).unwrap();

        // Lamellar has 1 parameter (a), shape is [[a]]
        // So J should have shape (1, 1, 1) with J[0,0,0] = ∂h[0,0]/∂a = 1
        assert_eq!(j.shape(), &[1, 1, 1]);
        assert_approx_eq!(f64, j[[0, 0, 0]], 1.0, epsilon = 1e-5);
    }

    #[test]
    fn test_jacobian_rectangular() {
        let jacobian = CellJacobian::new(1e-6);
        let cell = UnitCell::rectangular(10.0, 8.0).unwrap();

        let j = jacobian.compute(&cell).unwrap();

        // Rectangular has 2 parameters (a, b)
        // Shape is [[a, 0], [0, b]]
        // J[0,0,0] = ∂h[0,0]/∂a = 1
        // J[1,1,1] = ∂h[1,1]/∂b = 1
        // All others should be 0
        assert_eq!(j.shape(), &[2, 2, 2]);
        assert_approx_eq!(f64, j[[0, 0, 0]], 1.0, epsilon = 1e-5); // ∂h[0,0]/∂a
        assert_approx_eq!(f64, j[[1, 1, 1]], 1.0, epsilon = 1e-5); // ∂h[1,1]/∂b
        assert_approx_eq!(f64, j[[0, 1, 0]], 0.0, epsilon = 1e-5); // ∂h[0,1]/∂a
        assert_approx_eq!(f64, j[[0, 1, 1]], 0.0, epsilon = 1e-5); // ∂h[0,1]/∂b
    }

    #[test]
    fn test_jacobian_cubic() {
        let jacobian = CellJacobian::new(1e-6);
        let cell = UnitCell::cubic(10.0).unwrap();

        let j = jacobian.compute(&cell).unwrap();

        // Cubic has 1 parameter (a), shape is [[a, 0, 0], [0, a, 0], [0, 0, a]]
        // J[0,0,0] = J[1,1,0] = J[2,2,0] = 1 (diagonal elements)
        assert_eq!(j.shape(), &[3, 3, 1]);
        assert_approx_eq!(f64, j[[0, 0, 0]], 1.0, epsilon = 1e-5);
        assert_approx_eq!(f64, j[[1, 1, 0]], 1.0, epsilon = 1e-5);
        assert_approx_eq!(f64, j[[2, 2, 0]], 1.0, epsilon = 1e-5);
    }

    #[test]
    fn test_project_stress_lamellar() {
        let jacobian = CellJacobian::new(1e-6);
        let cell = UnitCell::lamellar(10.0).unwrap();

        let j = jacobian.compute(&cell).unwrap();
        let stress = array![[2.0]];

        let grad = jacobian.project_stress(&j, &stress);

        // grad[0] = Σ_{i,j} J[i,j,0] * stress[i,j] = J[0,0,0] * stress[0,0] = 1.0 * 2.0 = 2.0
        assert_eq!(grad.len(), 1);
        assert_approx_eq!(f64, grad[0], 2.0, epsilon = 1e-5);
    }

    #[test]
    fn test_project_stress_rectangular() {
        let jacobian = CellJacobian::new(1e-6);
        let cell = UnitCell::rectangular(10.0, 8.0).unwrap();

        let j = jacobian.compute(&cell).unwrap();
        let stress = array![[1.0, 0.0], [0.0, 2.0]];

        let grad = jacobian.project_stress(&j, &stress);

        // grad[0] = J[0,0,0] * stress[0,0] = 1.0 * 1.0 = 1.0
        // grad[1] = J[1,1,1] * stress[1,1] = 1.0 * 2.0 = 2.0
        assert_eq!(grad.len(), 2);
        assert_approx_eq!(f64, grad[0], 1.0, epsilon = 1e-5);
        assert_approx_eq!(f64, grad[1], 2.0, epsilon = 1e-5);
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
        let a = cell.parameters()[0];
        assert!(a > 0.0, "Cell parameter should remain positive");
    }

    #[test]
    fn test_update_cell_lamellar() {
        let updater = CellUpdater::new(1.0);
        let cell = UnitCell::lamellar(10.0).unwrap();
        let stress = array![[2.0]]; // Positive stress → decrease a

        let new_cell = updater.update_cell(&cell, &stress).unwrap();

        assert_eq!(new_cell.lattice(), CellLattice::Lamellar);
        let a = new_cell.parameters()[0];
        assert!(a < 10.0, "Parameter should decrease with positive stress");
        assert!(a > 0.0, "Parameter should remain positive");
    }

    #[test]
    fn test_update_cell_square() {
        let updater = CellUpdater::new(1.0);
        let cell = UnitCell::square(10.0).unwrap();
        let stress = array![[1.0, 0.0], [0.0, 1.0]]; // Equal diagonal stress

        let new_cell = updater.update_cell(&cell, &stress).unwrap();

        assert_eq!(new_cell.lattice(), CellLattice::Square);
        let a = new_cell.parameters()[0];
        // Jacobian: both diagonal elements contribute to 'a' parameter
        // grad = σ[0,0] + σ[1,1] = 1.0 + 1.0 = 2.0
        // a_new = 10 - 1.0 * 2.0 = 8.0
        assert_approx_eq!(f64, a, 8.0, epsilon = 1e-5);
    }

    #[test]
    fn test_update_cell_rectangular() {
        let updater = CellUpdater::new(1.0);
        let cell = UnitCell::rectangular(10.0, 8.0).unwrap();
        let stress = array![[2.0, 0.0], [0.0, 1.0]]; // Different diagonal stresses

        let new_cell = updater.update_cell(&cell, &stress).unwrap();

        assert_eq!(new_cell.lattice(), CellLattice::Rectangular);
        let values = new_cell.parameters();
        // Relaxed epsilon to account for finite difference numerical error
        assert_approx_eq!(f64, values[0], 8.0, epsilon = 1e-5); // 10 - 2
        assert_approx_eq!(f64, values[1], 7.0, epsilon = 1e-5); // 8 - 1
    }

    #[test]
    fn test_update_cell_cubic() {
        let updater = CellUpdater::new(1.0);
        let cell = UnitCell::cubic(10.0).unwrap();
        let stress = array![[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]];

        let new_cell = updater.update_cell(&cell, &stress).unwrap();

        assert_eq!(new_cell.lattice(), CellLattice::Cubic);
        let a = new_cell.parameters()[0];
        // Jacobian: all three diagonal elements contribute to 'a' parameter
        // grad = σ[0,0] + σ[1,1] + σ[2,2] = 1.0 + 2.0 + 3.0 = 6.0
        // a_new = 10 - 1.0 * 6.0 = 4.0
        assert_approx_eq!(f64, a, 4.0, epsilon = 1e-5);
    }

    #[test]
    fn test_update_cell_orthorhombic() {
        let updater = CellUpdater::new(1.0);
        let cell = UnitCell::orthorhombic(10.0, 8.0, 6.0).unwrap();
        let stress = array![[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]];

        let new_cell = updater.update_cell(&cell, &stress).unwrap();

        assert_eq!(new_cell.lattice(), CellLattice::Orthorhombic);
        let values = new_cell.parameters();
        // Relaxed epsilon to account for finite difference numerical error
        assert_approx_eq!(f64, values[0], 9.0, epsilon = 1e-5); // 10 - 1
        assert_approx_eq!(f64, values[1], 6.0, epsilon = 1e-5); // 8 - 2
        assert_approx_eq!(f64, values[2], 3.0, epsilon = 1e-5); // 6 - 3
    }

    #[test]
    fn test_hexagonal_cell_supported() {
        let updater = CellUpdater::new(1.0);
        let cell = UnitCell::hexagonal_2d(10.0).unwrap();
        let stress = array![[1.0, 0.0], [0.0, 1.0]];

        let result = updater.update_cell(&cell, &stress);
        assert!(result.is_ok(), "Hexagonal2D should now be supported");

        let new_cell = result.unwrap();
        assert_eq!(new_cell.lattice(), CellLattice::Hexagonal2D);
        let a = new_cell.parameters()[0];
        // Should average diagonal stress
        assert!(a < 10.0);
    }
}
