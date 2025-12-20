use std::collections::HashMap;

use super::{propagator::PropagatorDirection, Propagator, PropagatorStep, StepMethod};
use crate::{
    chem::Block,
    domain::{Domain, Mesh, FFT},
    fields::{FieldOps, RField},
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
    ds: f64,
}

impl BlockSolver {
    pub fn new(block: Block, mesh: Mesh, ns: usize, ds: f64) -> Self {
        let fft = crate::domain::FFT::new(mesh);
        let step = PropagatorStep::new(mesh);
        let forward = Propagator::new(mesh, ns);
        let reverse = Propagator::new(mesh, ns);
        let concentration = RField::zeros(mesh);
        Self {
            fft,
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

    /// Compute the stress contribution from this block.
    ///
    /// Uses Fredrickson's chain stretching formula:
    /// σ_ij = -(b²V_monomer/6V) Σ_k (g^{-1}k)_i (g^{-1}k)_j W(k)
    /// where W(k) = ∫ q_f(k,s) q_r(k,s) ds
    pub fn compute_stress(&mut self, domain: &Domain, phi: f64, partition: f64) -> Vec<f64> {
        use crate::fields::CField;

        let volume = domain.cell().volume();
        let kvecs = domain.kvecs();
        let metric_inv = domain.cell().metric_inv();
        let nk = kvecs.nrows();

        // Initialize stress tensor
        let ncomponents = domain.mesh().stress_components();
        let mut stress = vec![0.0; ncomponents];

        // Compute k-space transformed vectors
        let kvecs_transformed = kvecs.dot(metric_inv);

        // Prefactor: -(b²V_monomer φ) / (6V Q)
        let b_sq = self.block.segment_length * self.block.segment_length;
        let prefactor = -(phi * b_sq * self.block.monomer.size) / (6.0 * volume * partition);

        // Reuse FFT for transforming propagators
        let kmesh = self.mesh.kmesh();
        let mut qf_k = CField::zeros(kmesh);
        let mut qr_k = CField::zeros(kmesh);

        // Compute W(k) = ∫ q_f(k,s) q_r(k,s) ds
        let mut weights = vec![0.0; nk];
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

            // Accumulate q_f * q_r†
            for (ik, (&qf, &qr)) in qf_k.iter().zip(qr_k.iter()).enumerate() {
                weights[ik] += coef * (qf * qr.conj()).re;
            }
        }

        // Normalize integral
        for w in weights.iter_mut() {
            *w *= self.ds / 3.0;
        }

        // Compute stress tensor components
        for ik in 0..nk {
            let k = kvecs_transformed.row(ik);
            let w = weights[ik];

            match domain.mesh() {
                crate::domain::Mesh::One(_) => {
                    stress[0] += prefactor * w * k[0] * k[0];
                }
                crate::domain::Mesh::Two(_, _) => {
                    stress[0] += prefactor * w * k[0] * k[0]; // σ_xx
                    stress[1] += prefactor * w * k[1] * k[1]; // σ_yy
                    stress[2] += prefactor * w * k[0] * k[1]; // σ_xy
                }
                crate::domain::Mesh::Three(_, _, _) => {
                    stress[0] += prefactor * w * k[0] * k[0]; // σ_xx
                    stress[1] += prefactor * w * k[1] * k[1]; // σ_yy
                    stress[2] += prefactor * w * k[2] * k[2]; // σ_zz
                    stress[3] += prefactor * w * k[0] * k[1]; // σ_xy
                    stress[4] += prefactor * w * k[0] * k[2]; // σ_xz
                    stress[5] += prefactor * w * k[1] * k[2]; // σ_yz
                }
            }
        }

        stress
    }
}

#[cfg(test)]
mod tests {
    use float_cmp::assert_approx_eq;
    use ndarray_rand::{rand_distr::Normal, RandomExt};
    use rand::{rngs::SmallRng, SeedableRng};

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
