use fts::chem::{Block, Monomer, Point, Polymer, Species, SpeciesDescription};
use pyo3::prelude::*;

use crate::impl_conversions;

#[pyclass(name = "Monomer", module = "pyfts.chem", frozen)]
#[derive(Clone, Copy)]
pub struct PyMonomer {
    core: Monomer,
}

impl_conversions!(Monomer, PyMonomer);

#[pymethods]
impl PyMonomer {
    #[new]
    fn __new__(id: usize, size: f64) -> Self {
        Monomer::new(id, size).into()
    }

    #[getter]
    fn get_id(&self) -> usize {
        self.core.id
    }

    #[getter]
    fn get_size(&self) -> f64 {
        self.core.size
    }
}

#[pyclass(name = "Block", module = "pyfts.chem", frozen)]
#[derive(Clone, Copy)]
pub struct PyBlock {
    core: Block,
}

impl_conversions!(Block, PyBlock);

#[pymethods]
impl PyBlock {
    #[new]
    fn __new__(monomer: PyMonomer, repeat_units: usize, segment_length: f64) -> Self {
        Block::new(monomer.into(), repeat_units, segment_length).into()
    }

    #[getter]
    fn get_monomer(&self) -> PyMonomer {
        self.core.monomer.into()
    }

    #[getter]
    fn get_repeat_units(&self) -> usize {
        self.core.repeat_units
    }

    #[getter]
    fn get_segment_length(&self) -> f64 {
        self.core.segment_length
    }
}

#[pyclass(name = "Species", module = "pyfts.chem", subclass, frozen)]
#[derive(Clone)]
pub struct PySpecies {
    core: Species,
}

impl_conversions!(Species, PySpecies);

#[pymethods]
impl PySpecies {
    #[getter]
    fn get_phi(&self) -> f64 {
        self.core.phi()
    }

    #[getter]
    fn get_size(&self) -> f64 {
        self.core.size()
    }

    fn monomers(&self) -> Vec<PyMonomer> {
        self.core.monomers().iter().copied().map(|m| m.into()).collect()
    }

    fn monomer_fraction(&self, id: usize) -> f64 {
        self.core.monomer_fraction(id)
    }
}

#[pyclass(name = "Point", module = "pyfts.chem", extends=PySpecies, frozen)]
pub struct PyPoint {}

#[pymethods]
impl PyPoint {
    #[new]
    fn __new__(monomer: PyMonomer, phi: f64) -> (Self, PySpecies) {
        let point: Species = Point::new(monomer.core, phi).into();
        let base: PySpecies = point.into();
        (PyPoint {}, base)
    }
}

#[pyclass(name = "Polymer", module = "pyfts.chem", extends=PySpecies, frozen)]
pub struct PyPolymer {}

#[pymethods]
impl PyPolymer {
    #[new]
    fn __new__(blocks: Vec<PyBlock>, contour_steps: usize, phi: f64) -> (Self, PySpecies) {
        let blocks = blocks.into_iter().map(|b| b.into()).collect();
        let polymer: Species = Polymer::new(blocks, contour_steps, phi).into();
        let base: PySpecies = polymer.into();
        (PyPolymer {}, base)
    }

    #[getter]
    fn get_blocks(self_: PyRef<'_, Self>) -> Vec<PyBlock> {
        let super_ = self_.as_ref();
        match &super_.core {
            Species::Polymer(polymer) => polymer.blocks.iter().cloned().map(|b| b.into()).collect(),
            // Should not be reachable
            _ => panic!("PyPolymer didn't contain a Polymer species"),
        }
    }
}
