use std::collections::HashMap;

use enum_dispatch::enum_dispatch;

use super::{PointSolver, PolymerSolver};
use crate::{
    chem::{Monomer, Species},
    domain::{Domain, Mesh},
    RField,
};

#[derive(Debug)]
pub struct SolverState {
    pub partition: f64,
    pub density: HashMap<usize, RField>,
}

impl SolverState {
    pub fn new(mesh: Mesh, monomers: impl IntoIterator<Item = Monomer>) -> Self {
        Self {
            partition: 1.0,
            density: HashMap::from_iter(monomers.into_iter().map(|m| (m.id, RField::zeros(mesh)))),
        }
    }
}

#[enum_dispatch]
pub trait SolverOps {
    fn species(&self) -> Species;

    fn state(&self) -> &SolverState;

    fn solve(&mut self, domain: &Domain, fields: &[RField]);
}

#[enum_dispatch(SolverOps)]
#[derive(Debug)]
pub enum SpeciesSolver {
    PointSolver,
    PolymerSolver,
}

impl SpeciesSolver {
    pub fn new(mesh: Mesh, species: Species) -> Self {
        match species {
            Species::Point(point) => PointSolver::new(mesh, point).into(),
            Species::Polymer(polymer) => PolymerSolver::new(mesh, polymer).into(),
        }
    }
}
