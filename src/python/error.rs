use pyo3::{exceptions::PyException, PyResult};

use crate::Result;

pub struct ToPyResult<T>(pub Result<T>);

impl<T> From<ToPyResult<T>> for PyResult<T> {
    fn from(v: ToPyResult<T>) -> Self {
        v.0.map_err(|e| PyException::new_err(format!("{}", e)))
    }
}

impl<T> ToPyResult<T> {
    pub fn into_py(self) -> PyResult<T> {
        self.into()
    }
}
