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
        let partition_sum = Self::partition_sum(self.forward.head(), self.reverse.tail());
        partition_sum / self.mesh.size() as f64
    }

    fn partition_sum(q1: &RField, q2: &RField) -> f64 {
        Zip::from(q1).and(q2).fold(0.0, |acc, h, t| acc + h * t)
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
        let ns = self.ns();
        self.concentration.fill(0.0);

        // Reverse to account for N - s indexing of qfields
        let forward_iter = self.forward.iter();
        let reverse_iter = self.reverse.iter().rev();

        forward_iter
            .zip(reverse_iter)
            .enumerate()
            .for_each(|(s, (qf, qr))| {
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
                Zip::from(&mut self.concentration)
                    .and(qf)
                    .and(qr)
                    .for_each(|c, f, r| *c += coef * f * r);
            });

        // Normalize the integral
        self.concentration *= prefactor * (self.ds / 3.0);
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

        let mut domain = Domain::new(mesh, cell).unwrap();
        domain.update_ksq();

        let mut rng = SmallRng::seed_from_u64(0);
        let distr = Normal::new(0.0, 0.1).unwrap();
        let field = RField::random_using(mesh, &distr, &mut rng);
        let ksq = domain.ksq();

        let block = Block::new(Monomer::new(0, 1.0), 100, 1.0);
        let mut solver = BlockSolver::new(block, mesh, 10, 0.1);
        solver.update_step(&vec![field], ksq);
        solver.solve(None, PropagatorDirection::Forward);
        solver.solve(None, PropagatorDirection::Reverse);

        solver
    }

    #[test]
    fn test_propagator_symmetry() {
        let solver = get_solver();
        let qf = solver.forward();
        let qr = solver.reverse();

        // Heads of both propagators should be equal to 1
        assert!(qf.head().iter().all(|x| *x == 1.0));
        assert!(qr.head().iter().all(|x| *x == 1.0));

        // Propagators should be equivalent at all contour points due to symmetry
        for (f, r) in qf.iter().zip(qr.iter()) {
            assert_eq!(f, r);
        }
    }

    #[test]
    fn test_partition_symmetry() {
        let solver = get_solver();
        let qf = solver.forward();
        let qr = solver.reverse();

        // Partition product should be equivalent at all (s, N-s-1) pairs along the chain
        let partition_sums: Vec<f64> = qf
            .iter()
            .zip(qr.iter().rev())
            .map(|(f, r)| BlockSolver::partition_sum(f, r))
            .collect();

        let first = partition_sums[0];
        partition_sums
            .into_iter()
            .for_each(|elem| assert_approx_eq!(f64, elem, first));
    }
}
