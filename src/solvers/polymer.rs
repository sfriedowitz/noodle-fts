use super::{solver::SolverOps, BlockPropagator, SolverInput, SolverState};
use crate::{chem::Polymer, domain::Mesh};

fn build_propagators(polymer: &Polymer) -> Vec<BlockPropagator> {
    todo!()
}

#[derive(Debug)]
pub struct PolymerSolver {
    polymer: Polymer,
    state: SolverState,
    forward_propagators: Vec<BlockPropagator>,
    reverse_propagators: Vec<BlockPropagator>,
}

impl PolymerSolver {
    pub fn new(polymer: Polymer, mesh: Mesh) -> Self {
        let state = SolverState::new(mesh, polymer.monomer_ids());
        let forward_propagators = build_propagators(&polymer);
        let reverse_propagators = forward_propagators.iter().rev().cloned().collect();
        Self {
            polymer,
            state,
            forward_propagators,
            reverse_propagators,
        }
    }
}

impl SolverOps for PolymerSolver {
    fn state(&self) -> &SolverState {
        &self.state
    }

    fn update_state<'a>(&mut self, input: &SolverInput<'a>) {
        todo!()
    }
}
