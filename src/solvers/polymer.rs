use super::{BlockSolver, PropagatorDirection, SolverOps, SolverState};
use crate::{
    chem::{Polymer, Species, SpeciesDescription},
    domain::{Domain, Mesh},
    fields::RField,
};

#[derive(Debug)]
pub struct PolymerSolver {
    polymer: Polymer,
    state: SolverState,
    solvers: Vec<BlockSolver>,
}

impl PolymerSolver {
    pub fn new(polymer: Polymer, mesh: Mesh) -> Self {
        let mut state = SolverState::default();
        for monomer in polymer.monomers() {
            state.density.insert(monomer.id, RField::zeros(mesh));
        }
        let solvers = Self::build_block_solvers(&polymer, mesh);
        Self {
            polymer,
            state,
            solvers,
        }
    }

    fn build_block_solvers(polymer: &Polymer, mesh: Mesh) -> Vec<BlockSolver> {
        let target_ds = polymer.size() / polymer.contour_steps as f64;
        polymer
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
        self.polymer.clone().into()
    }

    fn state(&self) -> &SolverState {
        &self.state
    }

    fn solve<'a>(&mut self, domain: &Domain, fields: &[RField]) {
        // Get ksq grid from domain
        let ksq = domain.ksq();

        // Update solver steps with current fields
        for solver in self.solvers.iter_mut() {
            solver.update_step(&fields, &ksq);
        }

        // Propagate forward (initialize first solver w/ empty source)
        let mut source: Option<&RField> = None;
        for solver in self.solvers.iter_mut() {
            solver.solve(source, PropagatorDirection::Forward);
            source = Some(solver.forward().tail());
        }

        // Propagate reverse (initialize last solver w/ empty source)
        source = None;
        for solver in self.solvers.iter_mut().rev() {
            solver.solve(source, PropagatorDirection::Reverse);
            source = Some(solver.reverse().tail());
        }

        // Compute new partition function
        let partition_sum = self.solvers.last().unwrap().forward().tail().sum();
        self.state.partition = partition_sum / domain.mesh_size() as f64;

        // Update solver density
        let prefactor = self.polymer.fraction / self.polymer.size() / self.state.partition;
        for solver in self.solvers.iter_mut() {
            solver.update_density(prefactor);
        }

        // Accumulate per-block density in solver state
        for (id, rho) in self.state.density.iter_mut() {
            rho.fill(0.0);
            for solver in self.solvers.iter() {
                if solver.block().monomer.id == *id {
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
        let field_a = field
            .mapv(|f| (1.0 - 2.0 * f.cosh().powf(-2.0)) / n as f64)
            .into_dyn();
        let field_b = -1.0 * &field_a;
        let fields = vec![field_a, field_b];

        let monomer_a = Monomer::new(0, 1.0);
        let monomer_b = Monomer::new(1, 1.0);
        let block_a = Block::new(monomer_a, n, b);
        let block_b = Block::new(monomer_b, n, b);
        let polymer = Polymer::new(vec![block_a, block_b], ns, 1.0);

        let mesh = Mesh::One(nx);
        let cell = UnitCell::lamellar(length).unwrap();
        let domain = Domain::new(mesh, cell).unwrap();

        let mut solver = PolymerSolver::new(polymer, mesh);

        let now = Instant::now();
        solver.solve(&domain, &fields);
        let elapsed = now.elapsed();

        dbg!(x.as_slice().unwrap());
        dbg!(solver.state().density.get(&0).unwrap().as_slice().unwrap());
    }
}
