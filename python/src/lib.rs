mod chem;
mod system;

use pyo3::prelude::*;

// PyFTS Module
#[pymodule]
#[pyo3(name = "_pyfts")]
fn pyfts(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<chem::PyMonomer>()?;
    Ok(())
}
