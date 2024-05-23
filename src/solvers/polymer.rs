use super::{solver::SolverOps, BlockPropagator, SolverInput, SolverState};
use crate::{
    chem::{Polymer, Species, SpeciesDescription},
    domain::Mesh,
};

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

        todo!()
    }
}

impl SolverOps for PolymerSolver {
    fn species(&self) -> Species {
        self.polymer.clone().into()
    }

    fn state(&self) -> &SolverState {
        &self.state
    }

    fn update_state<'a>(&mut self, input: &SolverInput<'a>) {
        todo!()
    }
}
