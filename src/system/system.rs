use itertools::{izip, Itertools};
use rand::{distributions::Distribution, Rng};

use super::Interaction;
use crate::{
    chem::{Monomer, Species, SpeciesDescription},
    domain::{Domain, Mesh, UnitCell},
    fields::{FieldOps, RField},
    solvers::{SolverOps, SpeciesSolver},
    Result,
};

#[derive(Debug)]
pub struct System {
    // Components
    domain: Domain,
    interaction: Interaction,
    monomers: Vec<Monomer>,
    solvers: Vec<SpeciesSolver>,
    // State
    fields: Vec<RField>,
    concentrations: Vec<RField>,
    residuals: Vec<RField>,
    potentials: Vec<RField>,
    incompressibility: RField,
    total_concentration: RField,
}

impl System {
    pub fn new(mesh: Mesh, cell: UnitCell, species: Vec<Species>) -> Result<Self> {
        let monomers: Vec<Monomer> = species
            .iter()
            .flat_map(|s| s.monomers())
            .unique_by(|m| m.id)
            .sorted_by(|a, b| a.id.cmp(&b.id))
            .collect();

        let monomer_ids: Vec<usize> = monomers.iter().map(|m| m.id).collect();
        for (idx, monomer_id) in monomer_ids.iter().enumerate() {
            if idx != *monomer_id {
                return Err(format!("monomer IDs out of order {:?}", monomer_ids).into());
            }
        }

        let domain = Domain::new(mesh, cell)?;
        let interaction = Interaction::new(monomers.len());
        let solvers = species
            .into_iter()
            .map(|species| SpeciesSolver::new(mesh, species))
            .collect();

        let mesh = domain.mesh();
        let fields = vec![RField::zeros(domain.mesh()); monomers.len()];
        let concentrations = vec![RField::zeros(domain.mesh()); monomers.len()];
        let residuals = vec![RField::zeros(domain.mesh()); monomers.len()];
        let potentials = vec![RField::zeros(domain.mesh()); monomers.len()];
        let incompressibility = RField::zeros(mesh);
        let total_concentration = RField::zeros(mesh);

        Ok(System {
            domain,
            interaction,
            monomers,
            solvers,
            fields,
            concentrations,
            residuals,
            potentials,
            incompressibility,
            total_concentration,
        })
    }

    pub fn nmonomer(&self) -> usize {
        self.monomers.len()
    }

    pub fn nspecies(&self) -> usize {
        self.solvers.len()
    }

    pub fn monomers(&self) -> Vec<Monomer> {
        self.monomers.clone()
    }

    pub fn species(&self) -> Vec<Species> {
        self.solvers.iter().map(|s| s.species()).collect()
    }

    pub fn domain(&self) -> &Domain {
        &self.domain
    }

    pub fn interaction(&self) -> &Interaction {
        &self.interaction
    }

    pub fn interaction_mut(&mut self) -> &mut Interaction {
        &mut self.interaction
    }

    pub fn fields(&self) -> &[RField] {
        &self.fields
    }

    pub fn concentrations(&self) -> &[RField] {
        &self.concentrations
    }

    pub fn total_concentration(&self) -> &RField {
        &self.total_concentration
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

    pub fn iter<'a>(&'a self) -> impl Iterator<Item = MonomerState<'a>> {
        izip!(&self.fields, &self.concentrations, &self.residuals)
            .enumerate()
            .map(|(id, (field, concentration, residual))| MonomerState {
                id,
                field,
                concentration,
                residual,
            })
    }

    pub fn iter_mut<'a>(&'a mut self) -> impl Iterator<Item = MonomerStateMut<'a>> {
        izip!(&mut self.fields, &self.concentrations, &self.residuals)
            .enumerate()
            .map(|(id, (field, concentration, residual))| MonomerStateMut {
                id,
                field,
                concentration,
                residual,
            })
    }

    pub fn assign_fields(&mut self, fields: &[RField]) -> Result<()> {
        if fields.len() != self.nmonomer() {
            return Err("number of fields != number of monomers".into());
        }
        for (current, new) in izip!(&mut self.fields, fields) {
            if new.shape() != current.shape() {
                return Err("new field shape != system mesh shape".into());
            }
            current.assign(&new);
        }
        Ok(())
    }

    pub fn assign_concentrations(&mut self, concentrations: &[RField]) -> Result<()> {
        if concentrations.len() != self.nmonomer() {
            return Err("number of concentration fields != number of monomers".into());
        }
        for (current, new) in izip!(&mut self.concentrations, concentrations) {
            if new.shape() != current.shape() {
                return Err("new concentration shape != system mesh shape".into());
            }
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
            field.mapv_inplace(|_| distr.sample(rng));
        }
        self.update();
    }

    pub fn guess_fields(&mut self) {
        for field in self.fields.iter_mut() {
            field.fill(0.0);
        }
        self.interaction
            .add_gradients(&self.concentrations, &mut self.fields);
        self.update()
    }

    /// Update the system state.
    ///
    /// This involves:
    /// - Updating the wave-vectors based on the current cell parameters.
    /// - Solving the concentrations for all species.
    /// - Updating the Lagrange multiplier incompressibility field.
    /// - Updating the system residuals based on the current fields and concentrations.
    pub fn update(&mut self) {
        self.update_concentrations();
        self.update_potentials();
        self.update_incompressibility();
        self.update_residuals();
    }

    fn update_concentrations(&mut self) {
        // Reset concentration fields
        self.total_concentration.fill(0.0);
        for conc in self.concentrations.iter_mut() {
            conc.fill(0.0);
        }
        // Solve the species given current fields/domain
        let ksq = self.domain.ksq();
        for solver in self.solvers.iter_mut() {
            solver.solve(&self.fields, &ksq);
            for (id, conc) in solver.concentrations().iter() {
                self.concentrations[*id] += conc;
                self.total_concentration += conc;
            }
        }
    }

    fn update_potentials(&mut self) {
        for potential in self.potentials.iter_mut() {
            potential.fill(0.0);
        }
        self.interaction
            .add_gradients(&self.concentrations, &mut self.potentials)
    }

    fn update_incompressibility(&mut self) {
        // Initial incompressibility set to ctot - 1.0
        self.incompressibility
            .zip_mut_with(&self.total_concentration, |z, ctot| *z = ctot - 1.0);

        // Add w - p for all fields
        for (field, potential) in izip!(&self.fields, &self.potentials) {
            self.incompressibility
                .zip_mut_with_two(field, potential, |z, w, p| *z += w - p);
        }

        // Average over nmonomer
        self.incompressibility /= self.nmonomer() as f64;
    }

    fn update_residuals(&mut self) {
        // Residuals set to r = z + p - w
        for (residual, field, potential) in izip!(&mut self.residuals, &self.fields, &self.potentials) {
            residual.assign(&self.incompressibility);
            *residual += potential;
            *residual -= field;
        }
        // Mean-subtract fields and residuals (only closed ensemble)
        for (field, residual) in izip!(&mut self.fields, &mut self.residuals) {
            *residual -= residual.mean().unwrap();
            *field -= field.mean().unwrap();
        }
    }

    /// Return the Helmholtz free energy based on the current system state.
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
        let f_exchange: f64 = izip!(&self.fields, &self.concentrations)
            .map(|(field, conc)| {
                let exchange_sum = field.fold_with(conc, 0.0, |acc, w, c| acc - w * c);
                exchange_sum / self.domain.mesh().size() as f64
            })
            .sum();

        // Interaction
        let f_inter = self.interaction.energy(&self.concentrations);

        f_trans + f_exchange + f_inter
    }

    /// Return the Helmholtz free energy for the homogenous mixture.
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

    /// Return the weighted error based on the current field residuals.
    ///
    /// Reference: https://link.springer.com/article/10.1140/epje/i2009-10534-3
    pub fn field_error(&self) -> f64 {
        let mut rsum = 0.0;
        let mut wsum = 0.0;
        for state in self.iter() {
            rsum += state.residual.fold(0.0, |acc, r| acc + r * r);
            wsum += state.field.fold(0.0, |acc, w| acc + w * w);
        }
        (rsum / wsum).sqrt()
    }
}

#[derive(Debug)]
pub struct MonomerState<'a> {
    pub id: usize,
    pub field: &'a RField,
    pub concentration: &'a RField,
    pub residual: &'a RField,
}

#[derive(Debug)]
pub struct MonomerStateMut<'a> {
    pub id: usize,
    pub field: &'a mut RField,
    pub concentration: &'a RField,
    pub residual: &'a RField,
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

        let monomer_a = Monomer::new(0, 1.0);
        let monomer_b = Monomer::new(1, 1.0);

        let block = Block::new(monomer_a, 100, 1.0);
        let polymer = Polymer::new(vec![block], 100, 0.5);
        let point = Point::new(monomer_b, 0.5);
        let species: Vec<Species> = vec![polymer.into(), point.into()];

        // When: System initialized with zero fields (concentration is equal to bulk values)
        let mut system = System::new(mesh, cell, species).unwrap();
        system.interaction_mut().set_chi(0, 1, 0.25);
        system.update();

        // Then: Free energy should be equal to its bulk value
        let f = system.free_energy();
        let f_bulk = system.free_energy_bulk();
        assert_approx_eq!(f64, f, f_bulk);
    }
}
