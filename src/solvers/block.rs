use std::collections::HashMap;

use ndarray::{Array1, Array2, Axis};

use super::{Propagator, PropagatorStep, StepMethod, propagator::PropagatorDirection};
use crate::{
    chem::Block,
    domain::{Domain, FFT, Mesh},
    fields::{CField, FieldOps, RField},
};

#[derive(Debug)]
pub struct BlockSolver {
    fft: FFT,
    mesh: Mesh,
    block: Block,
    step: PropagatorStep,
    forward: Propagator,
    reverse: Propagator,
    concentration: RField,
    stress: Array2<f64>,
    ds: f64,
}

impl BlockSolver {
    pub fn new(block: Block, mesh: Mesh, ns: usize, ds: f64) -> Self {
        let fft = crate::domain::FFT::new(mesh);
        let step = PropagatorStep::new(mesh);
        let forward = Propagator::new(mesh, ns);
        let reverse = Propagator::new(mesh, ns);
        let concentration = RField::zeros(mesh);
        let stress = Array2::zeros((mesh.ndim(), mesh.ndim()));
        Self {
            fft,
            mesh,
            block,
            step,
            forward,
            reverse,
            concentration,
            stress,
            ds,
        }
    }

    pub fn ns(&self) -> usize {
        self.forward.ns()
    }

    pub fn ds(&self) -> f64 {
        self.ds
    }

    pub fn block(&self) -> &Block {
        &self.block
    }

    pub fn monomer_id(&self) -> usize {
        self.block.monomer.id
    }

    pub fn concentration(&self) -> &RField {
        &self.concentration
    }

    pub fn stress(&self) -> &Array2<f64> {
        &self.stress
    }

    pub fn forward(&self) -> &Propagator {
        &self.forward
    }

    pub fn reverse(&self) -> &Propagator {
        &self.reverse
    }

    pub fn compute_partition(&self, s: usize) -> f64 {
        let qf = self.forward.position(s);
        let qr = self.reverse.position(self.ns() - s - 1);
        let partition_sum = qf.fold_with(qr, 0.0, |acc, f, r| acc + f * r);
        partition_sum / self.mesh.size() as f64
    }

    pub fn solve(&mut self, source: Option<&RField>, direction: PropagatorDirection, method: StepMethod) {
        match direction {
            PropagatorDirection::Forward => {
                self.forward.update_head(source.into_iter());
                self.forward.propagate(&mut self.fft, &mut self.step, method);
            }
            PropagatorDirection::Reverse => {
                self.reverse.update_head(source.into_iter());
                self.reverse.propagate(&mut self.fft, &mut self.step, method);
            }
        }
    }

    pub fn update_step(&mut self, fields: &HashMap<usize, RField>, ksq: &RField) {
        let field = &fields.get(&self.block.monomer.id).unwrap();
        self.step.update_operators(
            field,
            ksq,
            self.block.segment_length,
            self.block.monomer.size,
            self.ds,
        );
    }

    pub fn update_concentration(&mut self, prefactor: f64) {
        let ns = self.ns();
        self.concentration.fill(0.0);

        // Reverse to account for N - s indexing of qfields
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
            self.concentration.zip_mut_with_two(
                self.forward.position(s),
                self.reverse.position(ns - s - 1),
                |c, f, r| *c += coef * f * r,
            );
        }

        // Normalize the integral
        self.concentration *= prefactor * (self.ds / 3.0);
    }

    pub fn update_stress(&mut self, domain: &Domain, phi: f64, partition: f64) {
        // TODO: This may not be correct. Validate prefactors?
        // TODO: Can any of this be done during propagator updates?
        // TODO: Can we get rid of any extra allocations here?
        let volume = domain.cell().volume();
        let kvecs = domain.kvecs().dot(&domain.cell().metric_inv());
        let nk = kvecs.nrows();

        // Prefactor: -(b²V_monomer φ) / (6V Q)
        let b_sq = self.block.segment_length * self.block.segment_length;
        let prefactor = -(phi * b_sq * self.block.monomer.size) / (6.0 * volume * partition);

        // Reuse FFT for transforming propagators
        let kmesh = self.mesh.kmesh();
        let mut qf_k = CField::zeros(kmesh);
        let mut qr_k = CField::zeros(kmesh);

        // Compute W(k) = ∫ q_f(k,s) q_r(k,s) ds
        let mut weights = Array1::zeros(nk);
        let ns = self.ns();

        for s in 0..ns {
            let qf_r = self.forward.position(s);
            let qr_r = self.reverse.position(ns - s - 1);

            // Transform to k-space
            self.fft.forward(qf_r, &mut qf_k);
            self.fft.forward(qr_r, &mut qr_k);

            // Simpson's rule coefficient
            let coef = if s == 0 || s == ns - 1 {
                1.0
            } else if s % 2 == 0 {
                2.0
            } else {
                4.0
            };

            // Accumulate q_f * q_r† using iterator chaining
            for ((w, &qf), &qr) in weights.iter_mut().zip(qf_k.iter()).zip(qr_k.iter()) {
                *w += coef * (qf * qr.conj()).re;
            }
        }

        // Normalize integral and apply prefactor
        weights *= prefactor * self.ds / 3.0;

        // Update stress tensor: σ_ij = Σ_k k_i k_j W(k) = K^T * diag(weights) * K
        // Use broadcasting to scale rows: need (nk, 1) shape for row-wise scaling
        let kvecs_weighted = &kvecs * &weights.insert_axis(Axis(1));
        self.stress = kvecs.t().dot(&kvecs_weighted);
    }
}

#[cfg(test)]
mod tests {
    use float_cmp::assert_approx_eq;
    use ndarray_rand::{RandomExt, rand_distr::Normal};
    use rand::{SeedableRng, rngs::SmallRng};

    use super::*;
    use crate::{
        chem::Monomer,
        domain::{Domain, UnitCell},
    };

    fn get_solver() -> BlockSolver {
        let mesh = Mesh::One(16);
        let cell = UnitCell::lamellar(10.0).unwrap();
        let domain = Domain::new(mesh, cell).unwrap();

        let mut rng = SmallRng::seed_from_u64(0);
        let distr = Normal::new(0.0, 0.1).unwrap();
        let field = RField::random_using(mesh, &distr, &mut rng);

        let fields = [(0, field)].into();
        let ksq = domain.ksq();

        let block = Block::new(Monomer::new(0, 1.0), 100, 1.0);
        let mut solver = BlockSolver::new(block, mesh, 10, 0.1);
        solver.update_step(&fields, &ksq);
        solver.solve(None, PropagatorDirection::Forward, StepMethod::RK2);
        solver.solve(None, PropagatorDirection::Reverse, StepMethod::RK2);

        solver
    }

    #[test]
    fn test_propagator_symmetry() {
        let solver = get_solver();

        // Heads of both propagators should be equal to 1
        assert!(solver.forward().head().iter().all(|x| *x == 1.0));
        assert!(solver.reverse().head().iter().all(|x| *x == 1.0));

        // Propagators should be equivalent at all contour points due to symmetry
        for s in 0..solver.ns() {
            let qf = solver.forward().position(s);
            let qr = solver.reverse().position(s);
            assert_eq!(qf, qr);
        }
    }

    #[test]
    fn test_partition_at_position() {
        let solver = get_solver();

        // Partition should be the same regardless of the contour index
        let partitions: Vec<f64> = (0..solver.ns()).map(|s| solver.compute_partition(s)).collect();

        partitions
            .iter()
            .for_each(|elem| assert_approx_eq!(f64, *elem, partitions[0]));
    }
}
