use fts::domain::{Mesh, UnitCell};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyTuple};

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
    #[pyo3(signature = (*dimensions))]
    fn __new__(dimensions: &Bound<'_, PyTuple>) -> PyResult<Self> {
        let dimensions: Vec<usize> = dimensions.extract()?;
        let mesh = match dimensions[..] {
            [] => Err(PyValueError::new_err("dimensions are empty")),
            [nx] => Ok(Mesh::One(nx)),
            [nx, ny] => Ok(Mesh::Two(nx, ny)),
            [nx, ny, nz] => Ok(Mesh::Three(nx, ny, nz)),
            _ => Err(PyValueError::new_err("more than 3 dimensions provided")),
        };
        mesh.map(|m| Self::new(m))
    }

    #[getter]
    fn get_size(&self) -> usize {
        self.core.size()
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
