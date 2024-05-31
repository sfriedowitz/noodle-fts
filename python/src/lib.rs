mod chem;
mod domain;
mod error;
mod utils;

use pyo3::prelude::*;

// PyFTS Module
#[pymodule]
#[pyo3(name = "_core")]
fn pyfts(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Chem
    m.add_class::<chem::PyMonomer>()?;
    m.add_class::<chem::PyBlock>()?;
    m.add_class::<chem::PySpecies>()?;
    m.add_class::<chem::PyPoint>()?;
    m.add_class::<chem::PyPolymer>()?;
    // Domain
    m.add_class::<domain::PyMesh>()?;
    m.add_class::<domain::PyUnitCell>()?;
    m.add_class::<domain::PyLamellarCell>()?;
    Ok(())
}
