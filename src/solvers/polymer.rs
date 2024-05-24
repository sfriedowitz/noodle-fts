use super::{
    block::{BlockSolver, PropagatorDirection},
    SolverOps, SolverState,
};
use crate::{
    chem::{Monomer, Polymer, Species, SpeciesDescription},
    domain::{Domain, Mesh},
    fields::RField,
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
        for monomer in polymer.monomers() {
            state.density.insert(monomer.id, RField::zeros(mesh));
        }

        let block_solvers = Self::build_block_solvers(&polymer, mesh);

        Self {
            polymer,
            state,
            block_solvers,
        }
    }

    fn build_block_solvers(polymer: &Polymer, mesh: Mesh) -> Vec<BlockSolver> {
        // Create block solvers and link consecutive pairs
        let mut block_solvers: Vec<BlockSolver> = polymer
            .blocks
            .iter()
            .copied()
            .map(|b| BlockSolver::new(b, mesh, 100, 1.0)) // TODO: Real discretize here
            .collect();

        for idx in 1..block_solvers.len() {
            let (head, tail) = block_solvers.split_at_mut(idx);
            let predecessor = head.last_mut().unwrap();
            let successor = tail.first_mut().unwrap();
            successor.add_source(predecessor);
        }

        block_solvers
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
        let partition_sum = self.block_solvers[0].reverse_propagator().tail().sum();
        self.state.partition = partition_sum / domain.mesh_size() as f64;

        // Update solver density
        let prefactor = self.polymer.fraction / self.polymer.size() / self.state.partition;
        for solver in self.block_solvers.iter_mut() {
            solver.update_density(prefactor);
        }

        // Accumulate per-block density in solver state
        for (id, rho) in self.state.density.iter_mut() {
            rho.fill(0.0);
            for solver in self.block_solvers.iter() {
                if solver.block.monomer.id == *id {
                    *rho += solver.density();
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use ndarray::Array1;
    use ndarray_rand::{rand_distr::Normal, RandomExt};

    use crate::{
        chem::{Block, Monomer, Polymer},
        domain::{Domain, Mesh, UnitCell},
        fields::RField,
        solvers::{PolymerSolver, SolverOps},
    };

    #[test]
    fn test_polymer_solver() {
        let n = 100;
        let ns = 100;
        let b = 1.0;
        let rg = ((n as f64) * b * b / 6.0).sqrt();

        let length = 10.0 * rg;
        let nx = 128;

        let x = Array1::linspace(0.0, length, nx);

        let field = (3.0 * (&x - length / 2.0) / (2.0 * rg));
        let field = field.mapv(|f| (1.0 - 2.0 * f.cosh().powf(-2.0)) / n as f64);
        let field = field.into_dyn();
        let fields = vec![field];

        let monomer = Monomer::new(0, 1.0);
        let block = Block::new(monomer, n, b);
        let polymer = Polymer::new(vec![block], ns, 1.0);

        let mesh = Mesh::One(nx);
        let cell = UnitCell::lamellar(length).unwrap();
        let domain = Domain::new(mesh, cell).unwrap();

        let mut solver = PolymerSolver::new(polymer, mesh);

        let now = Instant::now();
        solver.solve(&domain, &fields, &[monomer]);
        let elapsed = now.elapsed();

        dbg!(x.as_slice().unwrap());
        dbg!(solver.state().density.get(&0).unwrap().as_slice().unwrap());
    }
}
