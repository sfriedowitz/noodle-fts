use crate::{
    chem::Species,
    domain::Domain,
    fields::RField,
    solvers::{SolverOps, SpeciesSolver},
};

#[derive(Debug, Clone, Copy)]
pub enum Ensemble {
    Open,
    Closed,
}

pub struct System {
    ensemble: Ensemble,
    domain: Domain,
    solvers: Vec<SpeciesSolver>,
    fields: Vec<RField>,
    density: Vec<RField>,
}

impl System {
    pub fn new(ensemble: Ensemble, domain: Domain, species: Vec<Species>) -> Self {
        todo!()
    }

    pub fn nmonomer(&self) -> usize {
        todo!()
    }

    pub fn nspecies(&self) -> usize {
        todo!()
    }

    pub fn species(&self) -> Vec<Species> {
        self.solvers.iter().map(|s| s.species()).collect()
    }

    pub fn fields(&self) -> &[RField] {
        &self.fields
    }

    pub fn fields_mut(&mut self) -> &mut [RField] {
        &mut self.fields
    }

    pub fn density(&self) -> &[RField] {
        &self.density
    }

    pub fn solve(&mut self) {
        // Solve each species
        for solver in self.solvers.iter_mut() {
            solver.solve(&self.domain, &self.fields)
        }
        // Accumulate solver states into system state
    }
}
