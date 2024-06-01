mod chem;
mod domain;
mod error;
mod simulation;
mod system;
mod utils;

use pyo3::prelude::*;

#[pymodule]
pub fn pyfts(m: &Bound<'_, PyModule>) -> PyResult<()> {
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
    // System
    m.add_class::<system::PySystem>()?;
    // Simulation
    m.add_class::<simulation::PySCFTConfig>()?;
    m.add_class::<simulation::PySCFTState>()?;
    m.add_function(wrap_pyfunction!(simulation::scft, m)?)?;
    Ok(())
}
