use itertools::Itertools;
use ndarray::Zip;
use rand::{distributions::Distribution, Rng};

use super::Interaction;
use crate::{
    chem::{Monomer, Species, SpeciesDescription},
    domain::Domain,
    solvers::{SolverOps, SpeciesSolver},
    system::SystemError,
    RField, Result,
};

#[derive(Debug)]
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
            .sorted_by(|a, b| a.id.cmp(&b.id))
            .collect();

        Self::validate_monomers(&interaction, &monomers)?;

        let solvers = species
            .into_iter()
            .map(|species| SpeciesSolver::new(domain.mesh(), species))
            .collect();

        let fields = vec![RField::zeros(domain.mesh()); monomers.len()];
        let density = vec![RField::zeros(domain.mesh()); monomers.len()];
        let residuals = vec![RField::zeros(domain.mesh()); monomers.len()];

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
            return Err(Box::new(SystemError::EmptySpecies));
        }
        Ok(())
    }

    fn validate_monomers(interaction: &Interaction, monomers: &[Monomer]) -> Result<()> {
        if interaction.nmonomer() != monomers.len() {
            return Err(Box::new(SystemError::NumMonomers));
        }

        let monomer_ids: Vec<usize> = monomers.iter().map(|m| m.id).collect();
        for (idx, monomer_id) in monomer_ids.iter().copied().enumerate() {
            if monomer_id != idx {
                return Err(Box::new(SystemError::NonConsecutiveIDs(monomer_ids)));
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
                fractions[m.id] += s.phi() * s.monomer_fraction(m.id);
            }
        }
        fractions
    }

    pub fn fields(&self) -> &[RField] {
        &self.fields
    }

    pub fn density(&self) -> &[RField] {
        &self.density
    }

    pub fn residuals(&self) -> &[RField] {
        &self.residuals
    }

    pub fn iter_updater(&mut self) -> impl Iterator<Item = (&mut RField, &RField)> {
        self.fields.iter_mut().zip(&self.residuals)
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
            .gradients(&self.density, &mut self.residuals);

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

        // Mean-subtract fields and residuals (only closed ensemble)
        for (residual, field) in self.residuals.iter_mut().zip(self.fields.iter_mut()) {
            *residual -= residual.mean().unwrap();
            *field -= field.mean().unwrap();
        }
    }

    pub fn assign_fields(&mut self, fields: &[RField]) -> Result<()> {
        if fields.len() != self.nmonomer() {
            return Err(Box::new(SystemError::NumMonomers));
        }
        for (current, new) in self.fields.iter_mut().zip(fields.iter()) {
            current.assign(&new);
        }
        Ok(())
    }

    pub fn sample_fields<D, R>(&mut self, distr: &D, rng: &mut R)
    where
        D: Distribution<f64>,
        R: Rng,
    {
        for field in self.fields.iter_mut() {
            field.iter_mut().for_each(|f| *f = distr.sample(rng));
        }
    }

    pub fn guess_fields(&mut self) {
        for field in self.fields.iter_mut() {
            field.fill(0.0);
        }
        self.interaction.gradients(&self.density, &mut self.fields);
    }

    pub fn assign_density(&mut self, density: &[RField]) -> Result<()> {
        if density.len() != self.nmonomer() {
            return Err(Box::new(SystemError::NumMonomers));
        }
        for (current, new) in self.density.iter_mut().zip(density.iter()) {
            current.assign(&new);
        }
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

    use float_cmp::assert_approx_eq;

    use super::*;
    use crate::{
        chem::{Block, Point, Polymer},
        domain::{Mesh, UnitCell},
    };

    #[test]
    fn test_free_energy() {
        let mesh = Mesh::Two(32, 32);
        let cell = UnitCell::square(10.0).unwrap();
        let domain = Domain::new(mesh, cell).unwrap();

        let monomer_a = Monomer::new(0, 1.0);
        let monomer_b = Monomer::new(1, 1.0);

        let block = Block::new(monomer_a, 100, 1.0);
        let polymer = Polymer::new(vec![block], 100, 0.5);
        let point = Point::new(monomer_b, 0.5);
        let species: Vec<Species> = vec![polymer.into(), point.into()];

        let mut itx = Interaction::new(2);
        itx.set_chi(0, 1, 0.25);

        // When: System initialized with zero fields (density is equal to bulk values)
        let mut system = System::new(domain, itx, species).unwrap();
        system.update();

        // Then: Free energy should be equal to its bulk value
        let f = system.free_energy();
        let f_bulk = system.free_energy_bulk();
        assert_approx_eq!(f64, f, f_bulk);
    }
}
