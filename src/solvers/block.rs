use std::{cell::RefCell, rc::Rc};

use super::{Propagator, PropagatorDirection, PropagatorStep};
use crate::{
    chem::{Block, Monomer},
    domain::Mesh,
    error::Result,
    fields::RField,
};

#[derive(Debug)]
pub struct BlockSolver {
    block: Block,
    step: PropagatorStep,
    forward: Rc<RefCell<Propagator>>,
    reverse: Rc<RefCell<Propagator>>,
    density: RField,
    step_size: f64,
}

impl BlockSolver {
    pub fn new(block: Block, mesh: Mesh, ngrid: usize, step_size: f64) -> Result<Self> {
        let step = PropagatorStep::new(mesh);
        let forward = Rc::new(RefCell::new(Propagator::new(mesh, ngrid)?));
        let reverse = Rc::new(RefCell::new(Propagator::new(mesh, ngrid)?));
        let density = RField::zeros(mesh);
        Ok(Self {
            block,
            step,
            forward,
            reverse,
            density,
            step_size,
        })
    }

    pub fn link_predecessor(&mut self, predecessor: &mut Self) {
        // Predecessor's forward is the source of our forward
        self.forward
            .borrow_mut()
            .set_source(predecessor.forward.clone());
        // Our reverse is the source of predecessor's reverse
        predecessor
            .reverse
            .borrow_mut()
            .set_source(self.reverse.clone());
    }

    pub fn update_step(&mut self, monomers: &[Monomer], fields: &[RField], ksq: &RField) {
        let monomer_id = self.block.monomer_id;
        let monomer_size = monomers[monomer_id].size;
        let field = &fields[monomer_id];

        self.step.update(
            field,
            ksq,
            self.step_size,
            monomer_size,
            self.block.segment_length,
        )
    }

    pub fn propagate(&mut self, direction: PropagatorDirection) {
        match direction {
            PropagatorDirection::Forward => self.forward.borrow_mut().solve(&mut self.step),
            PropagatorDirection::Reverse => self.reverse.borrow_mut().solve(&mut self.step),
        }
    }
}
