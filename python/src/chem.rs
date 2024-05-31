use fts::chem::{Block, Monomer};
use pyo3::prelude::*;

#[pyclass(module = "pyfts._core", name = "Monomer")]
#[derive(Clone, Copy)]
pub struct PyMonomer {
    core: Monomer,
}

impl PyMonomer {
    pub fn new(monomer: Monomer) -> Self {
        Self { core: monomer }
    }
}

#[pymethods]
impl PyMonomer {
    #[new]
    fn __new__(id: usize, size: f64) -> Self {
        Self::new(Monomer::new(id, size))
    }

    #[getter]
    fn get_id(&self) -> usize {
        self.core.id
    }

    #[setter]
    fn set_id(&mut self, id: usize) {
        self.core.id = id;
    }

    #[getter]
    fn get_size(&self) -> f64 {
        self.core.size
    }

    #[setter]
    fn set_size(&mut self, size: f64) {
        self.core.size = size;
    }
}

impl From<PyMonomer> for Monomer {
    fn from(value: PyMonomer) -> Self {
        value.core
    }
}

#[pyclass(module = "pyfts._core", name = "Block")]
#[derive(Clone, Copy)]
pub struct PyBlock {
    core: Block,
}

impl PyBlock {
    pub fn new(block: Block) -> Self {
        Self { core: block }
    }
}

#[pymethods]
impl PyBlock {
    #[new]
    fn __new__(monomer: PyMonomer, repeat_units: usize, segment_length: f64) -> Self {
        let block = Block::new(monomer.into(), repeat_units, segment_length);
        Self { core: block }
    }

    #[getter]
    fn get_monomer(&self) -> PyMonomer {
        PyMonomer::new(self.core.monomer)
    }

    #[setter]
    fn set_monomer(&mut self, monomer: PyMonomer) {
        self.core.monomer = monomer.into();
    }

    #[getter]
    fn get_repeat_units(&self) -> usize {
        self.core.repeat_units
    }

    #[setter]
    fn set_repeat_units(&mut self, repeat_units: usize) {
        self.core.repeat_units = repeat_units;
    }

    #[getter]
    fn get_segment_length(&self) -> f64 {
        self.core.segment_length
    }

    #[setter]
    fn set_segment_length(&mut self, segment_length: f64) {
        self.core.segment_length = segment_length;
    }
}

impl From<PyBlock> for Block {
    fn from(value: PyBlock) -> Self {
        value.core
    }
}
