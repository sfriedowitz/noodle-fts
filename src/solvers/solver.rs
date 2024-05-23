use std::collections::{HashMap, HashSet};

use enum_dispatch::enum_dispatch;

use super::{PointSolver, PolymerSolver};
use crate::{
    chem::{Monomer, Species},
    domain::Mesh,
    fields::RField,
};

#[enum_dispatch]
pub trait SolverOps {
    fn species(&self) -> Species;

    fn state(&self) -> &SolverState;

    fn solve<'a>(&mut self, input: &SolverInput<'a>);
}

#[enum_dispatch(SolverOps)]
#[derive(Debug)]
pub enum SpeciesSolver {
    PointSolver,
    PolymerSolver,
}

#[derive(Debug)]
pub struct SolverInput<'a> {
    pub monomers: &'a [Monomer],
    pub fields: &'a [RField],
    pub ksq: &'a RField,
}

#[derive(Debug)]
pub struct SolverState {
    pub partition: f64,
    pub density: HashMap<usize, RField>,
}

impl SolverState {
    pub fn new(mesh: Mesh, monomer_ids: HashSet<usize>) -> Self {
        let density = monomer_ids
            .into_iter()
            .map(|id| (id, RField::zeros(mesh)))
            .collect();

        Self {
            partition: 0.0,
            density,
        }
    }
}
