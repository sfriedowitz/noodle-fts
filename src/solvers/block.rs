use super::{propagator::PropagatorDirection, Propagator, PropagatorStep, StepMethod};
use crate::{
    chem::Block,
    domain::Mesh,
    fields::{FieldOps, RField},
};

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
        let propagator = match direction {
            PropagatorDirection::Forward => &mut self.forward,
            PropagatorDirection::Reverse => &mut self.reverse,
        };
        propagator.update_head(source.into_iter());
        propagator.propagate(&mut self.step, method);
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
        let ksq = domain.ksq();

        let block = Block::new(Monomer::new(0, 1.0), 100, 1.0);
        let mut solver = BlockSolver::new(block, mesh, 10, 0.1);
        solver.update_step(&vec![field], &ksq);
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
