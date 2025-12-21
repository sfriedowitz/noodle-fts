use numpy::{PyArray2, ToPyArray};
use pyo3::{PyClass, exceptions::PyValueError, prelude::*, types::PyTuple};

use super::error::ToPyResult;
use crate::{
    domain::{CellParametersVariant, Mesh, UnitCell},
    impl_py_conversions,
};

#[pyclass(name = "Mesh", module = "pynoodle._core", frozen)]
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

    fn __repr__(&self) -> String {
        match self.0 {
            Mesh::One(nx) => format!("Mesh({nx})"),
            Mesh::Two(nx, ny) => format!("Mesh({nx}, {ny})"),
            Mesh::Three(nx, ny, nz) => format!("Mesh({nx}, {ny}, {nz})"),
        }
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

#[pyclass(name = "UnitCell", module = "pynoodle._core", subclass)]
#[derive(Clone)]
pub struct PyUnitCell(UnitCell);

impl_py_conversions!(UnitCell, PyUnitCell);

impl PyUnitCell {
    pub fn into_subclass(self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let variant = self.0.variant();
        let base = PyClassInitializer::from(self);
        match variant {
            CellParametersVariant::Lamellar => Self::add_subclass(py, base, PyLamellarCell {}),
            CellParametersVariant::Square => Self::add_subclass(py, base, PySquareCell {}),
            CellParametersVariant::Hexagonal2D => Self::add_subclass(py, base, PyHexagonal2DCell {}),
            CellParametersVariant::Cubic => Self::add_subclass(py, base, PyCubicCell {}),
            _ => todo!(),
        }
    }

    fn add_subclass<S: PyClass<BaseType = Self>>(
        py: Python<'_>,
        base: PyClassInitializer<Self>,
        sub: S,
    ) -> PyResult<Py<PyAny>> {
        let init = base.add_subclass(sub);
        let obj = Py::new(py, init)?;
        Ok(obj.into_any())
    }
}

#[pymethods]
impl PyUnitCell {
    fn __repr__(&self) -> String {
        // TODO: No idea how to get the name of the subclass at runtime
        format!("{:?}({:?})", self.0.variant(), self.0.values())
    }

    #[getter]
    fn get_ndim(&self) -> usize {
        self.0.ndim()
    }

    #[getter]
    fn get_parameters(&self) -> Vec<f64> {
        self.0.values().to_vec()
    }

    #[getter]
    fn get_shape<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.0.shape().to_pyarray(py)
    }

    #[getter]
    fn get_metric<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.0.metric().to_pyarray(py)
    }
}

#[pyclass(name = "LamellarCell", module = "pynoodle._core", extends=PyUnitCell)]
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

#[pyclass(name = "SquareCell", module = "pynoodle._core", extends=PyUnitCell)]
pub struct PySquareCell {}

#[pymethods]
impl PySquareCell {
    #[new]
    fn __new__(a: f64) -> PyResult<(Self, PyUnitCell)> {
        let cell = ToPyResult(UnitCell::square(a)).into_py()?;
        let base: PyUnitCell = cell.into();
        Ok((Self {}, base))
    }
}

#[pyclass(name = "Hexagonal2DCell", module = "pynoodle._core", extends=PyUnitCell)]
pub struct PyHexagonal2DCell {}

#[pymethods]
impl PyHexagonal2DCell {
    #[new]
    fn __new__(a: f64) -> PyResult<(Self, PyUnitCell)> {
        let cell = ToPyResult(UnitCell::hexagonal_2d(a)).into_py()?;
        let base: PyUnitCell = cell.into();
        Ok((Self {}, base))
    }
}

#[pyclass(name = "CubicCell", module = "pynoodle._core", extends=PyUnitCell)]
pub struct PyCubicCell {}

#[pymethods]
impl PyCubicCell {
    #[new]
    fn __new__(a: f64) -> PyResult<(Self, PyUnitCell)> {
        let cell = ToPyResult(UnitCell::cubic(a)).into_py()?;
        let base: PyUnitCell = cell.into();
        Ok((Self {}, base))
    }
}
