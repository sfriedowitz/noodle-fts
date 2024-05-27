use itertools::Itertools;
use ndarray::Zip;
use ndarray_rand::{rand_distr::Normal, RandomExt};

use super::Interaction;
use crate::{
    chem::{Monomer, Species, SpeciesDescription},
    domain::Domain,
    solvers::{SolverOps, SpeciesSolver},
    system::SystemError,
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
    pub fn new(domain: Domain, interaction: Interaction, species: Vec<Species>) -> Result<Self> {
        Self::validate_species(&species)?;

        let monomers: Vec<Monomer> = species
            .iter()
            .flat_map(|s| s.monomers())
            .unique_by(|m| m.id)
            .sorted_by(|a, b| b.id.cmp(&a.id))
            .collect();

        Self::validate_monomers(&interaction, &monomers)?;

        let solvers = species
            .into_iter()
            .map(|species| SpeciesSolver::new(domain.mesh(), species))
            .collect();

        let mut fields = vec![];
        let mut density = vec![];
        let mut residuals = vec![];
        for _ in 0..monomers.len() {
            let mesh = domain.mesh();
            fields.push(RField::random(mesh, Normal::new(0.0, 0.1)?));
            density.push(RField::zeros(mesh));
            residuals.push(RField::zeros(mesh));
        }

        Ok(System {
            domain,
            interaction,
            monomers,
            solvers,
            fields,
            density,
            residuals,
        })
    }

    fn validate_species(species: &[Species]) -> Result<()> {
        if species.is_empty() {
            return Err(Box::new(SystemError::Validation(
                "system must contain at least one species".into(),
            )));
        }
        Ok(())
    }

    fn validate_monomers(interaction: &Interaction, monomers: &[Monomer]) -> Result<()> {
        let nmonomer = monomers.len();

        if interaction.nmonomer() != nmonomer {
            return Err(Box::new(SystemError::Validation(format!(
                "interaction invalid for {nmonomer} monomers"
            ))));
        }

        for (idx, m) in monomers.iter().enumerate() {
            if m.id != idx {
                return Err(Box::new(SystemError::Validation(
                    "monomer IDs must be consecutive from [0, # monomers)".into(),
                )));
            }
        }

        Ok(())
    }

    pub fn nspecies(&self) -> usize {
        self.solvers.len()
    }

    pub fn species(&self) -> Vec<Species> {
        self.solvers.iter().map(|s| s.species()).collect()
    }

    pub fn nmonomer(&self) -> usize {
        self.monomers.len()
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

    fn fields(&self) -> &[RField] {
        &self.fields
    }

    fn fields_mut(&mut self) -> &mut [RField] {
        &mut self.fields
    }

    fn density(&self) -> &[RField] {
        &self.density
    }

    fn residuals(&self) -> &[RField] {
        &self.residuals
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
        for residual in self.residuals.iter_mut() {
            residual.fill(0.0);
        }

        // Add gradient of interaction
        self.interaction
            .add_potentials(&self.density, &mut self.residuals);

        // Initial residual: potential - actual fields
        for (residual, field) in self.residuals.iter_mut().zip(self.fields.iter()) {
            Zip::from(residual).and(field).for_each(|r, w| *r = *r - w);
        }

        // Residuals for monomers [1, nmonomer) are differences from monomer 0
        let (residual_zero, residual_others) = self.residuals.split_at_mut(1);
        for other in residual_others.iter_mut() {
            *other -= &residual_zero[0];
        }

        // Residual for monomer 0 imposes incompressibility: sum(density) - 1.0
        residual_zero[0].fill(-1.0);
        for density in self.density.iter() {
            residual_zero[0] += density;
        }
    }

    pub fn set_fields(&mut self, fields: Vec<RField>) -> Result<()> {
        if fields.len() != self.nmonomer() {
            return Err(Box::new(SystemError::NumFields));
        }

        self.fields = fields;
        self.update_residuals();

        Ok(())
    }

    pub fn set_density(&mut self, density: Vec<RField>) -> Result<()> {
        if density.len() != self.nmonomer() {
            return Err(Box::new(SystemError::NumFields));
        }

        // Must update field guess after setting the density
        // or else the density will be overwritten on next system update
        self.density = density;
        self.interaction
            .add_potentials(&self.density, &mut self.fields);
        self.update_residuals();

        Ok(())
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
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_system_validation() {
        todo!()
    }
}
