use std::{ops::Mul, rc::Rc};

use ndarray::Zip;
use num::complex::Complex64;

use super::{propagator::PropagatorDirection, step::PropagatorStep, BlockPropagator};
use crate::{
    chem::{Block, Monomer},
    domain::{Mesh, FFT},
    fields::{CField, Field, RField},
};

#[derive(Debug)]
pub struct BlockSolver {
    block: Block,
    step: PropagatorStep,
    forward: Rc<BlockPropagator>,
    reverse: Rc<BlockPropagator>,
    density: RField,
    step_size: f64,
}

impl BlockSolver {
    pub fn new(block: Block, mesh: Mesh, ngrid: usize, step_size: f64) -> Self {
        let step = PropagatorStep::new(mesh);
        let forward = Rc::new(BlockPropagator::new(mesh, ngrid));
        let reverse = Rc::new(BlockPropagator::new(mesh, ngrid));
        let density = RField::zeros(mesh);
        Self {
            block,
            step,
            forward,
            reverse,
            density,
            step_size,
        }
    }

    pub fn set_source(&mut self, source: &Self) {
        todo!()
    }

    pub fn update_operators(&mut self, monomers: &[Monomer], fields: &[RField], ksq: &RField) {
        let monomer = &monomers[self.block.monomer_id];
        let field = &fields[self.block.monomer_id];
        let lw_coeff = self.step_size / 2.0;
        let lk_coeff = self.step_size * self.block.segment_length.powf(2.0) / 6.0;

        // Update w-operators:
        // Full = exp(-w * ds / 2)
        // Half = exp(-w * (ds / 2) / 2)
        Zip::from(&mut self.step.lw_full)
            .and(&mut self.step.lw_half)
            .and(field)
            .for_each(|full, half, w| {
                *full = (-lw_coeff * monomer.size * w).exp();
                *half = (-lw_coeff * monomer.size * w / 2.0).exp();
            });

        // Update k-operators:
        // Full = exp(-k^2 * b^2 * ds / 6)
        // Half = exp(-k^2 * b^2 * (ds / 2) / 6)
        Zip::from(&mut self.step.lk_full)
            .and(&mut self.step.lk_half)
            .and(ksq)
            .for_each(|full, half, ksq| {
                let ksq_complex: Complex64 = ksq.into();
                *full = (-lk_coeff * ksq_complex).exp();
                *half = (-lk_coeff * ksq_complex / 2.0).exp();
            });
    }

    pub fn solve(&mut self, direction: PropagatorDirection) {
        todo!()
    }
}
