use std::collections::HashMap;

use enum_dispatch::enum_dispatch;

use super::{PointSolver, PolymerSolver};
use crate::{chem::Species, domain::Mesh, fields::RField};

#[enum_dispatch]
pub trait SolverOps {
    fn species(&self) -> Species;

    fn partition(&self) -> f64;

    fn concentrations(&self) -> &HashMap<usize, RField>;

    fn solve(&mut self, fields: &HashMap<usize, RField>, ksq: &RField);
}

#[enum_dispatch(SolverOps)]
#[derive(Debug)]
pub enum SpeciesSolver {
    PointSolver,
    PolymerSolver,
}

impl SpeciesSolver {
    pub fn new(mesh: Mesh, species: impl Into<Species>) -> Self {
        match species.into() {
            Species::Point(point) => PointSolver::new(mesh, point).into(),
            Species::Polymer(polymer) => PolymerSolver::new(mesh, polymer).into(),
        }
    }
}
