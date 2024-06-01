use std::collections::HashMap;

use super::{BlockSolver, PropagatorDirection, SolverOps, StepMethod};
use crate::{
    chem::{Polymer, Species, SpeciesDescription},
    domain::Mesh,
    fields::RField,
};

#[derive(Debug)]
pub struct PolymerSolver {
    species: Polymer,
    block_solvers: Vec<BlockSolver>,
    concentrations: HashMap<usize, RField>,
    partition: f64,
}

impl PolymerSolver {
    pub fn new(mesh: Mesh, species: Polymer) -> Self {
        let block_solvers = Self::build_block_solvers(mesh, &species);
        let concentrations =
            HashMap::from_iter(species.monomers().iter().map(|m| (m.id, RField::zeros(mesh))));
        Self {
            species,
            block_solvers,
            concentrations,
            partition: 1.0,
        }
    }

    fn build_block_solvers(mesh: Mesh, species: &Polymer) -> Vec<BlockSolver> {
        let target_ds = species.size() / species.contour_steps as f64;
        species
            .blocks
            .iter()
            .cloned()
            .map(|b| {
                // ns is guaranteed to be at least 3 for each block,
                // which results in an even number of contour steps per block
                let ns = (b.size() / (2.0 * target_ds) + 0.5).floor() as usize;
                let ns = 2 * ns.max(1) + 1;
                let ds = b.size() / ((ns - 1) as f64);
                BlockSolver::new(b, mesh, ns, ds)
            })
            .collect()
    }
}

impl SolverOps for PolymerSolver {
    fn species(&self) -> Species {
        self.species.clone().into()
    }

    fn partition(&self) -> f64 {
        self.partition
    }

    fn concentrations(&self) -> &HashMap<usize, RField> {
        &self.concentrations
    }

    fn solve(&mut self, fields: &[RField], ksq: &RField) {
        // Update solver steps with current fields
        for solver in self.block_solvers.iter_mut() {
            solver.update_step(&fields, &ksq);
        }

        // Propagate forward (initialize first solver w/ empty source)
        let mut source: Option<&RField> = None;
        for solver in self.block_solvers.iter_mut() {
            solver.solve(source, PropagatorDirection::Forward, StepMethod::RK2);
            source = Some(solver.forward().tail());
        }

        // Propagate reverse (initialize last solver w/ empty source)
        source = None;
        for solver in self.block_solvers.iter_mut().rev() {
            solver.solve(source, PropagatorDirection::Reverse, StepMethod::RK2);
            source = Some(solver.reverse().tail());
        }

        // Compute new partition function (any solver/contour position is identical)
        self.partition = self.block_solvers[0].compute_partition(0);

        // Update solver concentration
        let prefactor = self.species.phi() / self.species.size() / self.partition;
        for solver in self.block_solvers.iter_mut() {
            solver.update_concentration(prefactor);
        }

        // Accumulate per-block concentration in solver state
        for (id, concentration) in self.concentrations.iter_mut() {
            concentration.fill(0.0);
            for solver in self.block_solvers.iter() {
                if solver.monomer_id() == *id {
                    *concentration += solver.concentration();
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use float_cmp::assert_approx_eq;
    use ndarray_rand::{rand_distr::Normal, RandomExt};
    use rand::{rngs::SmallRng, SeedableRng};

    use super::*;
    use crate::{
        chem::{Block, Monomer},
        domain::{Domain, UnitCell},
    };

    #[test]
    fn test_multiblock_partition() {
        let mesh = Mesh::One(16);
        let cell = UnitCell::lamellar(10.0).unwrap();
        let domain = Domain::new(mesh, cell).unwrap();

        let nmonomer = 5;
        let blocks: Vec<Block> = (0..nmonomer)
            .map(|id| Block::new(Monomer::new(id, 1.0), 50, 1.0))
            .collect();
        let polymer = Polymer::new(blocks, 100, 1.0);

        let mut rng = SmallRng::seed_from_u64(0);
        let distr = Normal::new(0.0, 0.1).unwrap();
        let fields: Vec<RField> = (0..nmonomer)
            .map(|_| RField::random_using(mesh, &distr, &mut rng))
            .collect();
        let ksq = domain.ksq();

        let mut solver = PolymerSolver::new(mesh, polymer);
        solver.solve(&fields, &ksq);

        // Partition should be identical for any block along the chain contour
        let partitions: Vec<f64> = solver
            .block_solvers
            .iter()
            .map(|s| s.compute_partition(0))
            .collect();

        partitions
            .iter()
            .for_each(|elem| assert_approx_eq!(f64, *elem, partitions[0], epsilon = 1e-8));
    }
}
