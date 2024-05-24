use super::{grid::ContourGrid, step::PropagatorStep, SolverOps, SolverState};
use crate::{
    chem::{Monomer, Polymer, Species, SpeciesDescription},
    domain::{Domain, Mesh},
    fields::RField,
    solvers::step::StepMethod,
};

#[derive(Debug)]
pub struct PolymerSolver {
    polymer: Polymer,
    state: SolverState,
    grid: ContourGrid,
    step: PropagatorStep,
    qforward: Vec<RField>,
    qreverse: Vec<RField>,
    density: RField,
}

impl PolymerSolver {
    pub fn new(polymer: Polymer, mesh: Mesh) -> Self {
        let mut state = SolverState::default();
        for monomer in polymer.monomers() {
            state.density.insert(monomer.id, RField::zeros(mesh));
        }

        let grid = ContourGrid::new(&polymer);
        let step = PropagatorStep::new(mesh);

        let qforward = (0..grid.ns()).map(|_| RField::zeros(mesh)).collect();
        let qreverse = (0..grid.ns()).map(|_| RField::zeros(mesh)).collect();
        let density = RField::zeros(mesh);

        Self {
            polymer,
            state,
            grid,
            step,
            qforward,
            qreverse,
            density,
        }
    }
}

impl SolverOps for PolymerSolver {
    fn species(&self) -> Species {
        self.polymer.clone().into()
    }

    fn state(&self) -> &SolverState {
        &self.state
    }

    fn solve<'a>(&mut self, domain: &Domain, fields: &[RField]) {
        // Get ksq grid from domain
        let ksq = domain.ksq().unwrap();

        // // Forward propagation
        // self.qforward[0].fill(1.0);
        // for block in self.grid.into_iter() {
        //     // Update step with block-specific fields before propagation
        //     let field = &fields[block.monomer_id];
        //     self.step.update(
        //         field,
        //         &ksq,
        //         block.monomer_size,
        //         block.segment_length,
        //         block.ds,
        //     );
        //     // Propagate for range of block
        //     for s in block.forward_range() {
        //         let (left, right) = self.qforward.split_at_mut(s);
        //         let q_in = &left[left.len() - 1];
        //         let mut q_out = &mut right[0];
        //         self.step.apply(q_in, q_out, StepMethod::RQM4);
        //     }
        // }

        // // Reverse propagation
        // self.qreverse[0].fill(1.0);
        // for block in self.grid.into_iter().rev() {
        //     // Update step with block-specific fields before propagation
        //     let field = &fields[block.monomer_id];
        //     self.step.update(
        //         field,
        //         &ksq,
        //         block.monomer_size,
        //         block.segment_length,
        //         block.ds,
        //     );
        //     // Propagate for range of block
        //     for s in block.forward_range() {
        //         let (left, right) = self.qreverse.split_at_mut(s);
        //         let q_in = &left[left.len() - 1];
        //         let mut q_out = &mut right[0];
        //         self.step.apply(q_in, q_out, StepMethod::RQM4);
        //     }
        // }

        todo!()
    }
}
