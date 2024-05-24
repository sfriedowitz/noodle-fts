use std::{
    borrow::Borrow,
    cell::{Ref, RefCell, RefMut},
    ptr::NonNull,
    rc::Rc,
};

use ndarray::Zip;

use super::propagator::{Propagator, PropagatorStep};
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
pub(super) struct BlockSolver {
    pub block: Block,
    step: PropagatorStep,
    forward: Propagator,
    reverse: Propagator,
    density: RField,
    ds: f64,
}

impl BlockSolver {
    pub fn new(block: Block, mesh: Mesh, ns: usize, ds: f64) -> Self {
        let step = PropagatorStep::new(mesh);
        let forward = Propagator::new(mesh, ns);
        let reverse = Propagator::new(mesh, ns);
        let density = RField::zeros(mesh);
        Self {
            block,
            step,
            forward,
            reverse,
            density,
            ds,
        }
    }

    pub fn density(&self) -> &RField {
        &self.density
    }

    pub fn forward_propagator(&self) -> &Propagator {
        &self.forward
    }

    pub fn reverse_propagator(&self) -> &Propagator {
        &self.reverse
    }

    pub fn add_source(&mut self, other: &mut Self) {
        // Others's forward is the source of this forward
        self.forward.add_source(&mut other.forward);

        // This reverse is the source of other's reverse
        other.reverse.add_source(&mut self.reverse);
    }

    pub fn propagate(&mut self, direction: PropagatorDirection) {
        match direction {
            PropagatorDirection::Forward => self.forward.propagate(&mut self.step),
            PropagatorDirection::Reverse => self.reverse.propagate(&mut self.step),
        }
    }

    pub fn update_step(&mut self, monomers: &[Monomer], fields: &[RField], ksq: &RField) {
        let monomer_id = self.block.monomer_id;
        let monomer_size = monomers[monomer_id].size;
        let field = &fields[monomer_id];

        self.step
            .update(field, ksq, monomer_size, self.block.segment_length, self.ds)
    }

    pub fn update_density(&mut self, prefactor: f64) {
        let qf = self.forward.qfields();
        let qr = self.reverse.qfields();

        let ns = self.forward_propagator().ns();

        // Simpson's rule integration of qf * qr
        self.density.fill(0.0);
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
            // We integrate at contour position `s` for both forward and reverse
            // because index 0 into the qr vector is for position Ns on the chain
            Zip::from(&mut self.density)
                .and(&qf[s])
                .and(&qr[s])
                .for_each(|rho, x, y| *rho += coef * x * y);
        }

        // Normalize the integral
        self.density *= prefactor * (self.ds / 3.0);
    }
}
