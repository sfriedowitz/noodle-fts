mod chem;
mod domain;
mod error;
mod system;
mod utils;

use pyo3::prelude::*;

#[pymodule]
pub fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
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
    m.add_class::<domain::PySquareCell>()?;
    m.add_class::<domain::PyHexagonal2DCell>()?;
    m.add_class::<domain::PyCubicCell>()?;
    // System
    m.add_class::<system::PySystem>()?;
    m.add_class::<system::PyFieldUpdater>()?;
    m.add_class::<system::PyCellUpdater>()?;
    Ok(())
}
