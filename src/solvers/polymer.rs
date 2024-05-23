use super::{solver::SolverOps, SolverInput, SolverState};
use crate::{
    chem::{Polymer, Species, SpeciesDescription},
    domain::Mesh,
};

#[derive(Debug)]
pub struct PolymerSolver {
    polymer: Polymer,
    state: SolverState,
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

    fn solve<'a>(&mut self, input: &SolverInput<'a>) {
        todo!()
    }
}
