use fts::domain::{Mesh, UnitCell};
use pyo3::prelude::*;

use crate::error::ToPyResult;

#[pyclass(module = "pyfts._core", name = "Mesh")]
#[derive(Clone, Copy)]
pub struct PyMesh {
    core: Mesh,
}

impl PyMesh {
    pub fn new(mesh: Mesh) -> Self {
        Self { core: mesh }
    }
}

#[pymethods]
impl PyMesh {
    #[new]
    fn __new__() -> PyResult<Self> {
        todo!()
    }
}

#[pyclass(module = "pyfts._core", name = "UnitCell", subclass)]
#[derive(Clone)]
pub struct PyUnitCell {
    core: UnitCell,
}

impl PyUnitCell {
    pub fn new(cell: UnitCell) -> Self {
        Self { core: cell }
    }
}

#[pymethods]
impl PyUnitCell {
    #[getter]
    fn get_parameters(&self) -> Vec<f64> {
        vec![]
    }
}

#[pyclass(module = "pyfts._core", name = "LamellarCell", extends=PyUnitCell)]
pub struct PyLamellarCell {}

#[pymethods]
impl PyLamellarCell {
    #[new]
    fn __new__(a: f64) -> PyResult<(Self, PyUnitCell)> {
        let cell = ToPyResult(UnitCell::lamellar(a)).into_py()?;
        let base = PyUnitCell::new(cell);
        Ok((Self {}, base))
    }
}
