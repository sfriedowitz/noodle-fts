use super::{BlockSolver, SolverOps, SolverState};
use crate::{
    chem::{Monomer, Polymer, Species, SpeciesDescription},
    domain::{Domain, Mesh},
    fields::RField,
    solvers::PropagatorDirection,
};

#[derive(Debug)]
pub struct PolymerSolver {
    polymer: Polymer,
    state: SolverState,
    block_solvers: Vec<BlockSolver>,
}

impl PolymerSolver {
    pub fn new(polymer: Polymer, mesh: Mesh) -> Self {
        let mut state = SolverState::default();
        for monomer_id in polymer.monomer_ids() {
            state.density.insert(monomer_id, RField::zeros(mesh));
        }

        // Create block solvers and link consecutive pairs
        let mut block_solvers: Vec<BlockSolver> = polymer
            .blocks
            .iter()
            .copied()
            .map(|b| BlockSolver::new(b, mesh, 10, 1.0)) // TODO: Real discretize here
            .collect();

        for idx in 1..block_solvers.len() {
            let (head, tail) = block_solvers.split_at_mut(idx);
            let predecessor = head.last_mut().unwrap();
            let successor = tail.first_mut().unwrap();
            successor.link_predecessor(predecessor);
        }

        Self {
            polymer,
            state,
            block_solvers,
        }
    }
}

impl SolverOps for PolymerSolver {
    fn species(&self) -> Species {
        self.polymer.clone().into()
    }

    fn state(&self) -> &SolverState {
        &self.state
    }

    fn solve<'a>(&mut self, domain: &Domain, fields: &[RField], monomers: &[Monomer]) {
        // Get ksq grid from domain
        let ksq = domain.ksq().unwrap();

        // Propagate forward (update operators before)
        for solver in self.block_solvers.iter_mut() {
            solver.update_step(&monomers, &fields, &ksq);
            solver.propagate(PropagatorDirection::Forward)
        }

        // Propagate reverse
        for solver in self.block_solvers.iter_mut().rev() {
            solver.propagate(PropagatorDirection::Reverse)
        }

        // Compute new partition function
        let partition_sum = self.block_solvers[0].reverse_ref().q_last().sum();
        self.state.partition = partition_sum / domain.mesh_size() as f64;

        // Get block density and accumulate in solver state
        let prefactor = self.polymer.fraction / self.polymer.size(&monomers) / self.state.partition;
        for (id, rho) in self.state.density.iter_mut() {
            rho.fill(0.0);
            for solver in self.block_solvers.iter() {
                if solver.block.monomer_id != *id {
                    continue;
                }
                let density = prefactor * solver.density();
                *rho += &density;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use ndarray_rand::{rand_distr::Normal, RandomExt};

    use super::PolymerSolver;
    use crate::{
        chem::{Block, Monomer, Polymer},
        domain::{Domain, Mesh, UnitCell},
        fields::RField,
        solvers::SolverOps,
    };

    #[test]
    fn test_polymer_solver() {
        let mesh = Mesh::One(128);
        let cell = UnitCell::lamellar(10.0).unwrap();
        let domain = Domain::new(mesh, cell).unwrap();
        let fields = vec![RField::random(mesh, Normal::new(0.0, 0.1).unwrap())];

        let monomer = Monomer::new(0, 1.0);
        let block = Block::new(monomer.id, 100, 1.0);
        let polymer = Polymer::new(vec![block], 100, 1.0);

        let mut solver = PolymerSolver::new(polymer, mesh);

        let now = Instant::now();
        solver.solve(&domain, &fields, &[monomer]);
        let elapsed = now.elapsed();

        dbg!(elapsed);
    }
}
