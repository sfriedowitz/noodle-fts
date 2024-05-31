use pyo3::prelude::*;

#[pyclass(module = "pyfts._pyfts", name = "Monomer")]
#[derive(Clone, Copy)]
pub struct PyMonomer {
    #[pyo3(get, set)]
    id: usize,
    #[pyo3(get, set)]
    size: f64,
}

#[pymethods]
impl PyMonomer {
    #[new]
    fn new(id: usize, size: f64) -> Self {
        Self { id, size }
    }
}
