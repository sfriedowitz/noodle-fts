use super::{solver::SolverOps, BlockPropagator, SolverInput, SolverState};
use crate::{chem::Polymer, domain::Mesh, error::Result};

#[derive(Debug)]
pub struct PolymerSolver {
    polymer: Polymer,
    state: SolverState,
    propagators: Vec<BlockPropagator>,
}

impl PolymerSolver {
    pub fn new(polymer: Polymer, mesh: Mesh) -> Self {
        todo!()
    }
}

impl SolverOps for PolymerSolver {
    fn state(&self) -> &SolverState {
        &self.state
    }

    fn update_state<'a>(&mut self, input: &SolverInput<'a>) -> Result<()> {
        todo!()
    }
}
