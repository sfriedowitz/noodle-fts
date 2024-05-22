use std::collections::HashMap;

use enum_dispatch::enum_dispatch;

use super::{PointSolver, PolymerSolver};
use crate::{chem::Monomer, domain::Mesh, fields::RField};

#[enum_dispatch]
pub trait SolverOps {
    fn state(&self) -> &SolverState;

    fn update_state<'a>(&mut self, input: &SolverInput<'a>);
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
    pub fn new(mesh: Mesh, monomer_ids: &[usize]) -> Self {
        let mut density = HashMap::new();
        for id in monomer_ids {
            density.insert(*id, RField::zeros(mesh));
        }
        Self {
            partition: 0.0,
            density,
        }
    }
}
