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
    block: Block,
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

    pub fn ns(&self) -> usize {
        self.forward.ns()
    }

    pub fn block(&self) -> Block {
        self.block
    }

    pub fn density(&self) -> &RField {
        &self.density
    }

    pub fn forward(&self) -> &Propagator {
        &self.forward
    }

    pub fn reverse(&self) -> &Propagator {
        &self.reverse
    }

    pub fn solve(&mut self, source: Option<&RField>, direction: PropagatorDirection) {
        let propagator = match direction {
            PropagatorDirection::Forward => &mut self.forward,
            PropagatorDirection::Reverse => &mut self.reverse,
        };
        propagator.update_head(source.into_iter());
        propagator.propagate(&mut self.step);
    }

    pub fn update_step(&mut self, fields: &[RField], ksq: &RField) {
        let field = &fields[self.block.monomer.id];
        self.step.update_operators(
            field,
            ksq,
            self.block.segment_length,
            self.block.monomer.size,
            self.ds,
        );
    }

    pub fn update_density(&mut self, prefactor: f64) {
        // Propagators
        let qf = &self.forward;
        let qr = &self.reverse;

        // Reset density buffer
        self.density.fill(0.0);

        // Simpson's rule integration of qf * qr
        let ns = self.ns();
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
            // because index 0 into the qr vector is for position N on the chain
            Zip::from(&mut self.density)
                .and(qf.position(s))
                .and(qr.position(s))
                .for_each(|rho, x, y| *rho += coef * x * y);
        }

        // Normalize the integral
        self.density *= prefactor * (self.ds / 3.0);
    }
}
