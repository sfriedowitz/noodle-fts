use ndarray::Zip;

use super::{Propagator, PropagatorStep};
use crate::{chem::Block, domain::Mesh, RField};

#[derive(Debug, Clone, Copy)]
pub enum PropagatorDirection {
    Forward,
    Reverse,
}

#[derive(Debug)]
pub struct BlockSolver {
    mesh: Mesh,
    block: Block,
    step: PropagatorStep,
    forward: Propagator,
    reverse: Propagator,
    concentration: RField,
    ds: f64,
}

impl BlockSolver {
    pub fn new(block: Block, mesh: Mesh, ns: usize, ds: f64) -> Self {
        let step = PropagatorStep::new(mesh);
        let forward = Propagator::new(mesh, ns);
        let reverse = Propagator::new(mesh, ns);
        let concentration = RField::zeros(mesh);
        Self {
            mesh,
            block,
            step,
            forward,
            reverse,
            concentration,
            ds,
        }
    }

    pub fn ns(&self) -> usize {
        self.forward.ns()
    }

    pub fn block(&self) -> Block {
        self.block
    }

    pub fn concentration(&self) -> &RField {
        &self.concentration
    }

    pub fn forward(&self) -> &Propagator {
        &self.forward
    }

    pub fn reverse(&self) -> &Propagator {
        &self.reverse
    }

    pub fn compute_partition(&self) -> f64 {
        let partition_sum = Zip::from(self.forward.head())
            .and(self.reverse.tail())
            .fold(0.0, |acc, h, t| acc + h * t);
        partition_sum / self.mesh.size() as f64
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

    pub fn update_concentration(&mut self, prefactor: f64) {
        let qf = &self.forward;
        let qr = &self.reverse;
        self.concentration.fill(0.0);

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
            // because index 0 on the reverse propagator is for position N on the chain
            Zip::from(&mut self.concentration)
                .and(qf.position(s))
                .and(qr.position(ns - s - 1))
                .for_each(|c, x, y| *c += coef * x * y);
        }

        // Normalize the integral
        self.concentration *= prefactor * (self.ds / 3.0);
    }
}
