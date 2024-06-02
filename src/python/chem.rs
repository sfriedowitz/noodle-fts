use pyo3::prelude::*;

use crate::{
    chem::{Block, Monomer, Point, Polymer, Species, SpeciesDescription},
    impl_py_conversions,
};

#[pyclass(name = "Monomer", module = "pynoodle._core", frozen)]
#[derive(Clone, Copy)]
pub struct PyMonomer(Monomer);

impl_py_conversions!(Monomer, PyMonomer);

#[pymethods]
impl PyMonomer {
    #[new]
    fn __new__(id: usize, size: f64) -> Self {
        Monomer::new(id, size).into()
    }

    #[getter]
    fn get_id(&self) -> usize {
        self.0.id
    }

    #[getter]
    fn get_size(&self) -> f64 {
        self.0.size
    }
}

#[pyclass(name = "Block", module = "pynoodle._core", frozen)]
#[derive(Clone, Copy)]
pub struct PyBlock(Block);

impl_py_conversions!(Block, PyBlock);

#[pymethods]
impl PyBlock {
    #[new]
    fn __new__(monomer: PyMonomer, repeat_units: usize, segment_length: f64) -> Self {
        Block::new(monomer.into(), repeat_units, segment_length).into()
    }

    #[getter]
    fn get_monomer(&self) -> PyMonomer {
        self.0.monomer.into()
    }

    #[getter]
    fn get_repeat_units(&self) -> usize {
        self.0.repeat_units
    }

    #[getter]
    fn get_segment_length(&self) -> f64 {
        self.0.segment_length
    }
}

#[pyclass(name = "Species", module = "pynoodle._core", subclass, frozen)]
#[derive(Clone)]
pub struct PySpecies(Species);

impl_py_conversions!(Species, PySpecies);

#[pymethods]
impl PySpecies {
    #[getter]
    fn get_phi(&self) -> f64 {
        self.0.phi()
    }

    #[getter]
    fn get_size(&self) -> f64 {
        self.0.size()
    }

    fn monomers(&self) -> Vec<PyMonomer> {
        self.0.monomers().iter().copied().map(|m| m.into()).collect()
    }

    fn monomer_fraction(&self, id: usize) -> f64 {
        self.0.monomer_fraction(id)
    }
}

#[pyclass(name = "Point", module = "pynoodle._core", extends=PySpecies, frozen)]
pub struct PyPoint {}

#[pymethods]
impl PyPoint {
    #[new]
    fn __new__(monomer: PyMonomer, phi: f64) -> (Self, PySpecies) {
        let point: Species = Point::new(monomer.0, phi).into();
        let base: PySpecies = point.into();
        (PyPoint {}, base)
    }
}

#[pyclass(name = "Polymer", module = "pynoodle._core", extends=PySpecies, frozen)]
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
        match &super_.0 {
            Species::Polymer(polymer) => polymer.blocks.iter().cloned().map(|b| b.into()).collect(),
            // Should not be reachable
            _ => panic!("PyPolymer does not contain a Polymer species"),
        }
    }
}
