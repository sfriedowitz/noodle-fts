use ndarray::Zip;

use super::Interaction;
use crate::{
    chem::{Monomer, Species, SpeciesDescription},
    domain::Domain,
    solvers::{SolverOps, SpeciesSolver},
    RField, Result,
};

pub struct System {
    domain: Domain,
    interaction: Interaction,
    monomers: Vec<Monomer>,
    solvers: Vec<SpeciesSolver>,
    fields: Vec<RField>,
    density: Vec<RField>,
    residuals: Vec<RField>,
}

impl System {
    pub fn nmonomer(&self) -> usize {
        self.monomers.len()
    }

    pub fn nspecies(&self) -> usize {
        self.solvers.len()
    }

    pub fn species(&self) -> Vec<Species> {
        self.solvers.iter().map(|s| s.species()).collect()
    }

    pub fn monomers(&self) -> Vec<Monomer> {
        self.monomers.clone()
    }

    pub fn monomer_fractions(&self) -> Vec<f64> {
        let mut fractions = vec![0.0; self.nmonomer()];
        for m in self.monomers().iter() {
            for s in self.species().iter() {
                fractions[m.id] += s.monomer_fraction(m.id);
            }
        }
        fractions
    }

    pub fn free_energy(&self) -> f64 {
        // Translational
        let f_trans: f64 = self
            .solvers
            .iter()
            .map(|solver| {
                let species = solver.species();
                let mu = (species.phi() / solver.partition()).ln();
                (species.phi() / species.size()) * (mu - 1.0)
            })
            .sum();

        // Exchange
        let f_exchange: f64 = self
            .fields
            .iter()
            .zip(self.density.iter())
            .map(|(field, density)| {
                let exchange_sum = Zip::from(field)
                    .and(density)
                    .fold(0.0, |acc, w, d| acc + w * d);
                -1.0 * exchange_sum / self.domain.mesh_size() as f64
            })
            .sum();

        // Interaction
        let f_inter = self.interaction.energy(&self.density);

        f_trans + f_exchange + f_inter
    }

    pub fn free_energy_bulk(&self) -> f64 {
        // Translational
        let f_trans: f64 = self
            .solvers
            .iter()
            .map(|solver| {
                let species = solver.species();
                let mu = species.phi().ln();
                (species.phi() / species.size()) * (mu - 1.0)
            })
            .sum();

        // Interaction
        let f_inter = self.interaction.energy_bulk(&self.monomer_fractions());

        f_trans + f_inter
    }

    pub fn update(&mut self) {
        // Update ksq grid
        self.domain.update_ksq();

        // Reset density grids
        for rho in self.density.iter_mut() {
            rho.fill(0.0);
        }

        for solver in self.solvers.iter_mut() {
            // Solve the species given current fields/domain
            solver.solve(&self.domain, &self.fields);

            // Accumulate solver density into system density
            for (id, rho) in solver.density().iter() {
                self.density[*id] += rho;
            }
        }

        // Update residuals
        self.update_residuals();
    }

    fn update_residuals(&mut self) {
        // Reset residuals
        for res in self.residuals.iter_mut() {
            res.fill(0.0);
        }

        // Add gradient of interaction
        self.interaction
            .add_potentials(&self.density, &mut self.residuals);

        // Initial residual: potential - actual fields
        for (residual, field) in self.residuals.iter_mut().zip(self.fields.iter()) {
            Zip::from(residual).and(field).for_each(|r, w| *r = *r - w);
        }

        // Residuals for monomers [1, nmonomer) are differences from monomer 0
        let (res_zero, res_others) = self.residuals.split_at_mut(1);
        for other in res_others.iter_mut() {
            *other -= &res_zero[0];
        }

        // Residual for monomer 0 imposes incompressibility: sum(density) - 1.0
        res_zero[0].fill(-1.0);
        for density in self.density.iter() {
            res_zero[0] += density;
        }
    }
}

#[derive(Debug)]
pub struct SystemBuilder {
    domain: Option<Domain>,
    interaction: Option<Interaction>,
    species: Vec<Species>,
}

impl SystemBuilder {
    pub fn new() -> Self {
        Self {
            domain: None,
            interaction: None,
            species: vec![],
        }
    }

    pub fn with_domain(mut self, domain: Domain) -> Self {
        self.domain = Some(domain);
        self
    }

    pub fn with_interaction(mut self, interaction: Interaction) -> Self {
        self.interaction = Some(interaction);
        self
    }

    pub fn with_species(mut self, species: impl Into<Species>) -> Self {
        self.species.push(species.into());
        self
    }

    pub fn build(self) -> Result<System> {
        todo!();
    }
}
