use crate::{
    chem::Species,
    domain::Domain,
    solvers::{SolverOps, SpeciesSolver},
    RField, Result,
};

#[derive(Debug, Clone, Copy)]
pub enum Ensemble {
    Open,
    Closed,
}

pub struct SystemState {
    pub fields: Vec<RField>,
    pub density: Vec<RField>,
    pub potentials: Vec<RField>,
    pub residuals: Vec<RField>,
    pub partitions: Vec<f64>,
}

pub struct System {
    ensemble: Ensemble,
    domain: Domain,
    state: SystemState,
    solvers: Vec<SpeciesSolver>,
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

    pub fn state(&self) -> &SystemState {
        &self.state
    }

    pub fn solve(&mut self) -> Result<()> {
        // Update ksq grid
        self.domain.update_ksq()?;

        // Solve each species
        for solver in self.solvers.iter_mut() {
            solver.solve(&self.domain, &self.state.fields)
        }

        // Accumulate solver states into system state

        Ok(())
    }

    pub fn free_energy(&self) -> f64 {
        todo!()
    }

    pub fn free_energy_bulk(&self) -> f64 {
        todo!()
    }
}
