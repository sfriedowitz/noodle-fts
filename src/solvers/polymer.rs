use std::collections::HashMap;

use super::{BlockSolver, PropagatorDirection, SolverOps};
use crate::{
    chem::{Polymer, Species, SpeciesDescription},
    domain::{Domain, Mesh},
    RField,
};

#[derive(Debug)]
pub struct PolymerSolver {
    species: Polymer,
    block_solvers: Vec<BlockSolver>,
    density: HashMap<usize, RField>,
    partition: f64,
}

impl PolymerSolver {
    pub fn new(mesh: Mesh, species: Polymer) -> Self {
        let block_solvers = Self::build_block_solvers(mesh, &species);
        let density = HashMap::from_iter(
            species
                .monomers()
                .iter()
                .map(|m| (m.id, RField::zeros(mesh))),
        );
        Self {
            species,
            block_solvers,
            density,
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

    fn density(&self) -> &HashMap<usize, RField> {
        &self.density
    }

    fn solve(&mut self, domain: &Domain, fields: &[RField]) {
        // Get ksq grid from domain
        let ksq = domain.ksq();

        // Update solver steps with current fields
        for solver in self.block_solvers.iter_mut() {
            solver.update_step(&fields, &ksq);
        }

        // Propagate forward (initialize first solver w/ empty source)
        let mut source: Option<&RField> = None;
        for solver in self.block_solvers.iter_mut() {
            solver.solve(source, PropagatorDirection::Forward);
            source = Some(solver.forward().tail());
        }

        // Propagate reverse (initialize last solver w/ empty source)
        source = None;
        for solver in self.block_solvers.iter_mut().rev() {
            solver.solve(source, PropagatorDirection::Reverse);
            source = Some(solver.reverse().tail());
        }

        // Compute new partition function
        self.partition = self.block_solvers[0].compute_partition();

        // Update solver density
        let prefactor = self.species.phi() / self.species.size() / self.partition;
        for solver in self.block_solvers.iter_mut() {
            solver.update_density(prefactor);
        }

        // Accumulate per-block density in solver state
        for (id, density) in self.density.iter_mut() {
            density.fill(0.0);
            for solver in self.block_solvers.iter() {
                if solver.block().monomer.id == *id {
                    *density += solver.density();
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use ndarray::Array1;

    use super::*;
    use crate::{
        chem::{Block, Monomer},
        domain::UnitCell,
    };

    #[test]
    fn test_homopolymer() {
        let n = 100;
        let contour_steps = 200;
        let b = 1.0;

        let rg = ((n as f64) * b * b / 6.0).sqrt();
        let length = 10.0 * rg;
        let nx = 64;

        let x = Array1::linspace(0.0, length, nx);
        let field = 3.0 * (&x - length / 2.0) / (2.0 * rg);
        let field = field
            .mapv(|f| (1.0 - 2.0 * f.cosh().powf(-2.0)) / n as f64)
            .into_dyn();
        let fields = vec![field];

        let monomer = Monomer::new(0, 1.0);
        let block = Block::new(monomer, n / 2, b);
        let polymer = Polymer::new(vec![block, block], contour_steps, 1.0);

        let mesh = Mesh::One(nx);
        let cell = UnitCell::lamellar(length).unwrap();

        let mut domain = Domain::new(mesh, cell).unwrap();
        domain.update_ksq();

        let mut solver = PolymerSolver::new(mesh, polymer);

        let now = Instant::now();
        solver.solve(&domain, &fields);
        let elapsed = now.elapsed();

        dbg!(x.as_slice().unwrap());
        dbg!(solver.density().get(&0).unwrap().as_slice().unwrap());
        dbg!(elapsed);
    }
}
