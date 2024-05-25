use std::collections::{HashMap, HashSet};

use enum_dispatch::enum_dispatch;

use super::{PointSolver, PolymerSolver};
use crate::{
    chem::Species,
    domain::{Domain, Mesh},
    fields::RField,
};

#[derive(Debug, Default)]
pub struct SolverState {
    pub partition: f64,
    pub density: HashMap<usize, RField>,
}

#[enum_dispatch]
pub trait SolverOps {
    fn species(&self) -> Species;

    fn state(&self) -> &SolverState;

    fn solve<'a>(&mut self, domain: &Domain, fields: &[RField]);
}

#[enum_dispatch(SolverOps)]
#[derive(Debug)]
pub enum SpeciesSolver {
    PointSolver,
    PolymerSolver,
}

impl SpeciesSolver {
    pub fn new(species: Species, mesh: Mesh) -> Self {
        match species {
            Species::Point(s) => PointSolver::new(s, mesh).into(),
            Species::Polymer(s) => PolymerSolver::new(s, mesh).into(),
        }
    }
}
