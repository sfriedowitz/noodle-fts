use ndarray_rand::rand_distr::Normal;
use numpy::IntoPyArray;
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};
use rand::{rngs::SmallRng, SeedableRng};

use super::{
    chem::PySpecies,
    domain::{PyMesh, PyUnitCell},
    error::ToPyResult,
};
use crate::{impl_py_conversions, system::System};

#[pyclass(name = "System", module = "pyfts")]
pub struct PySystem(pub(crate) System);

impl_py_conversions!(System, PySystem);

#[pymethods]
impl PySystem {
    #[new]
    fn __new__(mesh: PyMesh, cell: PyUnitCell, species: Vec<PySpecies>) -> PyResult<Self> {
        let system = System::new(
            mesh.into(),
            cell.into(),
            species.into_iter().map(|s| s.into()).collect(),
        );
        let system = ToPyResult(system).into_py()?;
        Ok(Self(system))
    }

    fn free_energy(&self) -> f64 {
        self.0.free_energy()
    }

    fn free_energy_bulk(&self) -> f64 {
        self.0.free_energy_bulk()
    }

    fn field_error(&self) -> f64 {
        self.0.field_error()
    }

    fn set_interaction(&mut self, i: usize, j: usize, chi: f64) {
        self.0.interaction_mut().set_chi(i, j, chi)
    }

    fn fields<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new_bound(py);
        for (id, field) in self.0.fields().iter().enumerate() {
            dict.set_item(id, field.clone().into_pyarray_bound(py))?;
        }
        Ok(dict)
    }

    fn concentrations<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new_bound(py);
        for (id, conc) in self.0.concentrations().iter().enumerate() {
            dict.set_item(id, conc.clone().into_pyarray_bound(py))?;
        }
        Ok(dict)
    }

    #[pyo3(signature = (*, scale=0.1, seed=None))]
    fn sample_fields(&mut self, scale: f64, seed: Option<u64>) -> PyResult<()> {
        if let Ok(distr) = Normal::new(0.0, scale) {
            let mut rng = seed
                .map(SmallRng::seed_from_u64)
                .unwrap_or(SmallRng::from_entropy());
            self.0.sample_fields(&distr, &mut rng);
            Ok(())
        } else {
            Err(PyValueError::new_err(format!("invalid scale parameter: {scale}")))
        }
    }
}
