use pyo3::prelude::*;

use super::system::PySystem;
use crate::{
    impl_py_conversions,
    simulation::{SCFTConfig, SCFTState, SCFT},
};

#[pyclass(name = "SCFTConfig", module = "pyfts", frozen)]
#[derive(Clone, Copy)]
pub struct PySCFTConfig(SCFTConfig);

impl_py_conversions!(SCFTConfig, PySCFTConfig);

#[pymethods]
impl PySCFTConfig {
    #[new]
    #[pyo3(signature = (*, steps=100, step_size=0.1, field_tolerance=1e-5))]
    fn __new__(steps: usize, step_size: f64, field_tolerance: f64) -> Self {
        let inner = SCFTConfig {
            steps,
            step_size,
            field_tolerance,
        };
        Self(inner)
    }
}

#[pyclass(name = "SCFTState", module = "pyfts", frozen)]
#[derive(Clone, Copy)]
pub struct PySCFTState(SCFTState);

impl_py_conversions!(SCFTState, PySCFTState);

#[pymethods]
impl PySCFTState {
    #[getter]
    fn get_step(&self) -> usize {
        self.0.step
    }

    #[getter]
    fn get_elapsed(&self) -> f64 {
        self.0.elapsed.as_secs_f64()
    }

    #[getter]
    fn get_is_converged(&self) -> bool {
        self.0.is_converged
    }

    #[getter]
    fn get_field_error(&self) -> f64 {
        self.0.field_error
    }

    #[getter]
    fn get_free_energy(&self) -> f64 {
        self.0.free_energy
    }

    #[getter]
    fn get_free_energy_bulk(&self) -> f64 {
        self.0.free_energy_bulk
    }
}

#[pyfunction]
#[pyo3(signature = (system, *, config=None))]
pub fn scft(system: &Bound<'_, PySystem>, config: Option<PySCFTConfig>) -> PyResult<PySCFTState> {
    let config = config.map(|c| c.into()).unwrap_or(SCFTConfig::default());
    let simulation = SCFT::new(config);
    let mut system = system.borrow_mut();
    let result = simulation.run(&mut system.0);
    Ok(result.into())
}
