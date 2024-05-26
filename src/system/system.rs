use super::Interaction;
use crate::{
    chem::{Monomer, Species},
    domain::Domain,
    solvers::{SolverOps, SpeciesSolver},
    RField, Result,
};

pub struct System {
    domain: Domain,
    interaction: Interaction,
    solvers: Vec<SpeciesSolver>,
    fields: Vec<RField>,
    density: Vec<RField>,
    residuals: Vec<RField>,
}

impl System {
    pub fn new(domain: Domain, interaction: Interaction, species: Vec<Species>) -> Self {
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

    pub fn monomers(&self) -> Vec<Monomer> {
        todo!()
    }

    pub fn free_energy(&self) -> f64 {
        todo!()
    }

    pub fn free_energy_bulk(&self) -> f64 {
        todo!()
    }

    pub fn solve(&mut self) -> Result<()> {
        // Update ksq grid
        self.domain.update_ksq()?;

        // Solve each species
        for solver in self.solvers.iter_mut() {
            solver.solve(&self.domain, &self.fields)
        }

        // Accumulate solver states into system state

        Ok(())
    }
}

pub struct SystemBuilder {}
