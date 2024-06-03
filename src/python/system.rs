use ndarray_rand::rand_distr::Normal;
use numpy::{IntoPyArray, PyReadonlyArrayDyn};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};
use rand::{rngs::SmallRng, SeedableRng};

use super::{
    chem::PySpecies,
    domain::{PyMesh, PyUnitCell},
    error::ToPyResult,
};
use crate::{
    impl_py_conversions,
    system::{FieldUpdater, System},
};

#[pyclass(name = "System", module = "pyfts._core")]
pub struct PySystem(System);

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

    #[getter]
    fn nmonomer(&self) -> usize {
        self.0.nmonomer()
    }

    #[getter]
    fn nspecies(&self) -> usize {
        self.0.nspecies()
    }

    #[getter]
    fn mesh(&self) -> PyMesh {
        self.0.domain().mesh().into()
    }

    #[getter]
    fn cell(&self, py: Python<'_>) -> PyResult<PyObject> {
        let py_cell: PyUnitCell = self.0.domain().cell().clone().into();
        py_cell.into_subclass(py)
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
        for (id, field) in self.0.fields().iter() {
            dict.set_item(id, field.clone().into_pyarray_bound(py))?;
        }
        Ok(dict)
    }

    fn concentrations<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new_bound(py);
        for (id, conc) in self.0.concentrations().iter() {
            dict.set_item(id, conc.clone().into_pyarray_bound(py))?;
        }
        Ok(dict)
    }

    fn set_field(&mut self, id: usize, field: PyReadonlyArrayDyn<'_, f64>) -> PyResult<()> {
        let fview = field.as_array();
        ToPyResult(self.0.assign_field(id, fview)).into_py()
    }

    fn set_concentration(&mut self, id: usize, concentration: PyReadonlyArrayDyn<'_, f64>) -> PyResult<()> {
        let cview = concentration.as_array();
        ToPyResult(self.0.assign_concentration(id, cview)).into_py()
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

#[pyclass(name = "FieldUpdater", module = "pyfts._core")]
pub struct PyFieldUpdater(FieldUpdater);

impl_py_conversions!(FieldUpdater, PyFieldUpdater);

#[pymethods]
impl PyFieldUpdater {
    #[new]
    fn __new__(system: PyRef<'_, PySystem>, step_size: f64) -> Self {
        let updater = FieldUpdater::new(&system.0, step_size);
        Self(updater)
    }

    fn step(&mut self, system: &Bound<'_, PySystem>) -> PyResult<()> {
        let system = &mut system.borrow_mut().0;
        ToPyResult(self.0.step(system)).into_py()
    }
}
