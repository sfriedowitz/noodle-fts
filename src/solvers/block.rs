use std::{
    borrow::Borrow,
    cell::{Ref, RefCell},
    rc::Rc,
};

use ndarray::Zip;

use super::{Propagator, PropagatorDirection, PropagatorStep};
use crate::{
    chem::{Block, Monomer},
    domain::Mesh,
    fields::RField,
    math::simpsons_product,
};

#[derive(Debug)]
pub struct BlockSolver {
    pub block: Block,
    step: PropagatorStep,
    step_size: f64,
    forward: Rc<RefCell<Propagator>>,
    reverse: Rc<RefCell<Propagator>>,
}

impl BlockSolver {
    pub fn new(block: Block, mesh: Mesh, ngrid: usize, step_size: f64) -> Self {
        let step = PropagatorStep::new(mesh);
        let forward = Rc::new(RefCell::new(Propagator::new(mesh, ngrid)));
        let reverse = Rc::new(RefCell::new(Propagator::new(mesh, ngrid)));
        Self {
            block,
            step,
            step_size,
            forward,
            reverse,
        }
    }

    pub fn forward_propagator(&self) -> Ref<Propagator> {
        (*self.forward).borrow()
    }

    pub fn reverse_propagator(&self) -> Ref<Propagator> {
        (*self.reverse).borrow()
    }

    pub fn link_predecessor(&mut self, predecessor: &mut Self) {
        // Predecessor's forward is the source of this forward
        self.forward
            .borrow_mut()
            .add_source(predecessor.forward.clone());
        // This reverse is the source of predecessor's reverse
        predecessor
            .reverse
            .borrow_mut()
            .add_source(self.reverse.clone());
    }

    pub fn propagate(&mut self, direction: PropagatorDirection) {
        match direction {
            PropagatorDirection::Forward => self.forward.borrow_mut().propagate(&mut self.step),
            PropagatorDirection::Reverse => self.reverse.borrow_mut().propagate(&mut self.step),
        }
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

    pub fn density(&self, prefactor: f64) -> RField {
        let qf = self.forward_propagator();
        let qf = qf.qfields();

        let qr = self.reverse_propagator();
        let qr = qr.qfields();

        let mut density = simpsons_product(&qf, &qf, Some(self.step_size));
        density *= prefactor;

        density
    }
}
