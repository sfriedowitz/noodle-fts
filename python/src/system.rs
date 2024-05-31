use fts::system::System;
use pyo3::prelude::*;

/// Some docs
#[pyclass(module = "pyfts._pyfts", name = "System")]
pub struct PySystem {
    system: System,
}

impl PySystem {
    fn new(system: System) -> Self {
        Self { system }
    }
}

#[pymethods]
impl PySystem {
    #[new]
    fn __new__() -> PyResult<Self> {
        todo!()
    }

    fn free_energy(&self) -> PyResult<f64> {
        todo!()
    }
}
