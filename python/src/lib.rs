mod chem;
mod domain;
mod error;

use pyo3::prelude::*;

// PyFTS Module
#[pymodule]
#[pyo3(name = "_core")]
fn pyfts(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<chem::PyMonomer>()?;
    m.add_class::<chem::PyBlock>()?;
    m.add_class::<domain::PyMesh>()?;
    m.add_class::<domain::PyLamellarCell>()?;
    Ok(())
}
