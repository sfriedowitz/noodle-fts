use numpy::{IntoPyArray, PyArray2};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyTuple};

use super::error::ToPyResult;
use crate::{
    domain::{Mesh, UnitCell},
    impl_py_conversions,
};

#[pyclass(name = "Mesh", module = "pyfts._core", frozen)]
#[derive(Clone, Copy)]
pub struct PyMesh(Mesh);

impl_py_conversions!(Mesh, PyMesh);

#[pymethods]
impl PyMesh {
    #[new]
    #[pyo3(signature = (*dimensions))]
    fn __new__(dimensions: &Bound<'_, PyTuple>) -> PyResult<Self> {
        let err = PyValueError::new_err("invalid number of dimensions");
        let dimensions: Vec<usize> = dimensions.extract()?;
        let mesh = match dimensions[..] {
            [] => Err(err),
            [nx] => Ok(Mesh::One(nx)),
            [nx, ny] => Ok(Mesh::Two(nx, ny)),
            [nx, ny, nz] => Ok(Mesh::Three(nx, ny, nz)),
            _ => Err(err),
        };
        mesh.map(Self::from)
    }

    #[getter]
    fn get_size(&self) -> usize {
        self.0.size()
    }

    #[getter]
    fn get_dimensions(&self) -> Vec<usize> {
        self.0.dimensions()
    }
}

#[pyclass(name = "UnitCell", module = "pyfts._core", subclass)]
#[derive(Clone)]
pub struct PyUnitCell(UnitCell);

impl_py_conversions!(UnitCell, PyUnitCell);

#[pymethods]
impl PyUnitCell {
    #[getter]
    fn get_parameters(&self) -> Vec<f64> {
        self.0.parameters().into()
    }

    #[getter]
    fn get_shape<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.0.shape().clone().into_pyarray_bound(py)
    }

    #[getter]
    fn get_metric<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.0.metric().clone().into_pyarray_bound(py)
    }
}

#[pyclass(name = "LamellarCell", module = "pyfts._core", extends=PyUnitCell)]
pub struct PyLamellarCell {}

#[pymethods]
impl PyLamellarCell {
    #[new]
    fn __new__(a: f64) -> PyResult<(Self, PyUnitCell)> {
        let cell = ToPyResult(UnitCell::lamellar(a)).into_py()?;
        let base: PyUnitCell = cell.into();
        Ok((Self {}, base))
    }
}
