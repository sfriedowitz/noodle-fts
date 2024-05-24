use std::{
    borrow::Borrow,
    cell::{Ref, RefCell, RefMut},
    rc::Rc,
};

use ndarray::Zip;

use super::{Propagator, PropagatorStep};
use crate::{
    chem::{Block, Monomer},
    domain::Mesh,
    fields::RField,
};

#[derive(Debug, Clone, Copy)]
pub enum PropagatorDirection {
    Forward,
    Reverse,
}

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

    pub fn forward_ref(&self) -> Ref<Propagator> {
        (*self.forward).borrow()
    }

    pub fn reverse_ref(&self) -> Ref<Propagator> {
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
        let mut propagator = match direction {
            PropagatorDirection::Forward => self.forward.borrow_mut(),
            PropagatorDirection::Reverse => self.reverse.borrow_mut(),
        };
        propagator.propagate(&mut self.step);
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

    pub fn density(&self) -> RField {
        let forward_propagator = self.forward_ref();
        let qf = forward_propagator.q_fields();

        let reverse_propagator = self.reverse_ref();
        let qr = reverse_propagator.q_fields();

        let ns = forward_propagator.ns();
        let mut density = RField::zeros(qf[0].shape());

        // Simpson's rule integration of qf * qr
        for s in 0..ns {
            let coef = if s == 0 || s == ns - 1 {
                // Endpoints
                1.0
            } else if s % 2 == 0 {
                // Even indices
                2.0
            } else {
                // Odd indices
                4.0
            };
            Zip::from(&mut density)
                .and(&qf[s])
                .and(&qr[s])
                .for_each(|out, x, y| *out += coef * x * y);
        }

        // Normalize the integral
        density *= self.step_size / 3.0;

        density
    }
}
