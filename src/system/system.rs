use std::collections::HashMap;

use rand::{Rng, distr::Distribution};

use super::Interaction;
use crate::{
    Error, Result,
    chem::{Monomer, Species, SpeciesDescription},
    domain::{Domain, Mesh, UnitCell},
    fields::{FieldOps, RField, RFieldView},
    solvers::{SolverOps, SpeciesSolver},
};

fn generate_fields(mesh: Mesh, ids: &[usize]) -> HashMap<usize, RField> {
    ids.iter().cloned().map(|id| (id, RField::zeros(mesh))).collect()
}

#[derive(Debug)]
pub struct System {
    // Components
    domain: Domain,
    interaction: Interaction,
    monomers: HashMap<usize, Monomer>,
    solvers: Vec<SpeciesSolver>,
    // State
    fields: HashMap<usize, RField>,
    concentrations: HashMap<usize, RField>,
    residuals: HashMap<usize, RField>,
    potentials: HashMap<usize, RField>,
    incompressibility: RField,
    total_concentration: RField,
    stress: Vec<f64>,
}

impl System {
    pub fn new(mesh: Mesh, cell: UnitCell, species: Vec<Species>) -> Result<Self> {
        let monomers: HashMap<usize, Monomer> = species
            .iter()
            .flat_map(|s| s.monomers())
            .map(|m| (m.id, m))
            .collect();
        let monomer_ids: Vec<usize> = monomers.keys().into_iter().cloned().collect();

        let domain = Domain::new(mesh, cell)?;
        let interaction = Interaction::new();
        let solvers = species
            .into_iter()
            .map(|species| SpeciesSolver::new(mesh, species))
            .collect();

        let mesh = domain.mesh();
        let fields = generate_fields(mesh, &monomer_ids);
        let concentrations = generate_fields(mesh, &monomer_ids);
        let residuals = generate_fields(mesh, &monomer_ids);
        let potentials = generate_fields(mesh, &monomer_ids);
        let incompressibility = RField::zeros(mesh);
        let total_concentration = RField::zeros(mesh);
        let stress = Vec::new();

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
            stress,
        })
    }

    pub fn nmonomer(&self) -> usize {
        self.monomers.len()
    }

    pub fn nspecies(&self) -> usize {
        self.solvers.len()
    }

    pub fn monomers(&self) -> &HashMap<usize, Monomer> {
        &self.monomers
    }

    pub fn species(&self) -> Vec<Species> {
        self.solvers.iter().map(|s| s.species()).collect()
    }

    pub fn domain(&self) -> &Domain {
        &self.domain
    }

    pub fn domain_mut(&mut self) -> &mut Domain {
        &mut self.domain
    }

    pub fn interaction(&self) -> &Interaction {
        &self.interaction
    }

    pub fn interaction_mut(&mut self) -> &mut Interaction {
        &mut self.interaction
    }

    pub fn fields(&self) -> &HashMap<usize, RField> {
        &self.fields
    }

    pub fn concentrations(&self) -> &HashMap<usize, RField> {
        &self.concentrations
    }

    pub fn stress(&self) -> &[f64] {
        &self.stress
    }

    pub fn total_concentration(&self) -> &RField {
        &self.total_concentration
    }

    pub fn monomer_fractions(&self) -> HashMap<usize, f64> {
        let mut fractions = HashMap::new();
        for id in self.monomers().keys().cloned() {
            for s in self.species().iter() {
                let frac = fractions.entry(id).or_insert(0.0);
                *frac += s.phi() * s.monomer_fraction(id);
            }
        }
        fractions
    }

    pub fn iter<'a>(&'a self) -> impl Iterator<Item = MonomerState<'a>> {
        self.fields.iter().map(|(id, field)| MonomerState {
            id: *id,
            field,
            concentration: &self.concentrations[id],
            residual: &self.residuals[id],
        })
    }

    pub fn iter_mut<'a>(&'a mut self) -> impl Iterator<Item = MonomerStateMut<'a>> {
        self.fields.iter_mut().map(|(id, field)| MonomerStateMut {
            id: *id,
            field,
            concentration: &self.concentrations[id],
            residual: &self.residuals[id],
        })
    }

    pub fn assign_field(&mut self, id: usize, new: RFieldView<'_>) -> Result<()> {
        let current = self.fields.get_mut(&id).ok_or(Error::UnknownId(id))?;
        if new.shape() != current.shape() {
            Err(Error::Shape(current.shape().to_owned(), new.shape().to_owned()))
        } else {
            current.assign(&new);
            Ok(())
        }
    }

    pub fn assign_concentration(&mut self, id: usize, new: RFieldView<'_>) -> Result<()> {
        let current = self.concentrations.get_mut(&id).ok_or(Error::UnknownId(id))?;
        if new.shape() != current.shape() {
            Err(Error::Shape(current.shape().to_owned(), new.shape().to_owned()))
        } else {
            current.assign(&new);
            Ok(())
        }
    }

    pub fn sample_fields<D, R>(&mut self, distr: &D, rng: &mut R)
    where
        D: Distribution<f64>,
        R: Rng,
    {
        for field in self.fields.values_mut() {
            field.mapv_inplace(|_| distr.sample(rng));
        }
        self.update();
    }

    pub fn guess_fields(&mut self) {
        for field in self.fields.values_mut() {
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
    /// - Updating the cached stress tensor.
    pub fn update(&mut self) {
        self.update_concentrations();
        self.update_potentials();
        self.update_incompressibility();
        self.update_residuals();
        self.update_stress();
    }

    fn update_concentrations(&mut self) {
        // Reset concentration fields
        self.total_concentration.fill(0.0);
        for conc in self.concentrations.values_mut() {
            conc.fill(0.0);
        }
        // Solve the species given current fields/domain
        let ksq = self.domain.ksq();
        for solver in self.solvers.iter_mut() {
            solver.solve(&self.fields, &ksq);
            for (id, conc) in solver.concentrations().iter() {
                self.total_concentration += conc;
                *self.concentrations.get_mut(id).unwrap() += conc;
            }
        }
    }

    fn update_potentials(&mut self) {
        for potential in self.potentials.values_mut() {
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
        for id in self.monomers.keys() {
            let field = &self.fields[id];
            let potential = &self.potentials[id];
            self.incompressibility
                .zip_mut_with_two(field, potential, |z, w, p| *z += w - p);
        }

        // Average over nmonomer
        self.incompressibility /= self.nmonomer() as f64;
    }

    fn update_residuals(&mut self) {
        // Residuals set to r = z + p - w
        for (id, residual) in self.residuals.iter_mut() {
            let field = &self.fields[id];
            let potential = &self.potentials[id];

            residual.assign(&self.incompressibility);
            *residual += potential;
            *residual -= field;
        }

        // Mean-subtract fields and residuals (only closed ensemble)
        for field in self.fields.values_mut() {
            *field -= field.mean().unwrap();
        }
        for residual in self.residuals.values_mut() {
            *residual -= residual.mean().unwrap();
        }
    }

    fn update_stress(&mut self) {
        // Initialize stress tensor (size determined by first solver)
        self.stress = self.solvers[0].stress(&self.domain);

        // Sum contributions from remaining species
        for solver in self.solvers.iter_mut().skip(1) {
            let solver_stress = solver.stress(&self.domain);
            for (i, &s) in solver_stress.iter().enumerate() {
                self.stress[i] += s;
            }
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
        let f_exchange: f64 = self.iter().fold(0.0, |f, state| {
            let exchange_sum = state
                .field
                .fold_with(state.concentration, 0.0, |acc, w, c| acc - w * c);
            f + exchange_sum / self.domain.mesh().size() as f64
        });

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

    #[test]
    fn test_stress_homogeneous() {
        let mesh = Mesh::One(32);
        let cell = UnitCell::lamellar(10.0).unwrap();

        let monomer_a = Monomer::new(0, 1.0);
        let monomer_b = Monomer::new(1, 1.0);

        let block = Block::new(monomer_a, 100, 1.0);
        let polymer = Polymer::new(vec![block], 100, 0.5);
        let point = Point::new(monomer_b, 0.5);
        let species: Vec<Species> = vec![polymer.into(), point.into()];

        // When: System initialized with zero fields (homogeneous state)
        let mut system = System::new(mesh, cell, species).unwrap();
        system.interaction_mut().set_chi(0, 1, 0.25);
        system.update();

        // Then: Stress should be computable
        let stress = system.stress();
        assert_eq!(stress.len(), 1); // Lamellar has 1 stress component

        // Stress magnitude should be small for homogeneous system
        assert!(stress[0].abs() < 1.0, "Stress = {:?}", stress);
    }

    #[test]
    fn test_stress_1d_polymer() {
        let mesh = Mesh::One(32);
        let cell = UnitCell::lamellar(10.0).unwrap();

        let monomer_a = Monomer::new(0, 1.0);
        let block = Block::new(monomer_a, 100, 1.0);
        let polymer = Polymer::new(vec![block], 100, 1.0);
        let species: Vec<Species> = vec![polymer.into()];

        // When: System initialized and solved
        let mut system = System::new(mesh, cell, species).unwrap();
        system.update();

        // Then: Should compute stress
        let stress = system.stress();
        assert_eq!(stress.len(), 1); // Lamellar 1D has 1 stress component
    }

    #[test]
    fn test_stress_2d_symmetry() {
        let mesh = Mesh::Two(16, 16);
        let cell = UnitCell::square(10.0).unwrap();

        let monomer_a = Monomer::new(0, 1.0);
        let block = Block::new(monomer_a, 100, 1.0);
        let polymer = Polymer::new(vec![block], 100, 1.0);
        let species: Vec<Species> = vec![polymer.into()];

        // When: 2D system
        let mut system = System::new(mesh, cell, species).unwrap();
        system.update();

        // Then: Should return 3 components [σ_xx, σ_yy, σ_xy]
        let stress = system.stress();
        assert_eq!(stress.len(), 3);

        // For square cell with isotropic polymer, σ_xx should equal σ_yy
        assert_approx_eq!(f64, stress[0], stress[1], epsilon = 1e-6);

        // σ_xy should be zero by symmetry
        assert!(stress[2].abs() < 1e-10, "σ_xy = {}", stress[2]);
    }

    #[test]
    fn test_stress_point_particle() {
        let mesh = Mesh::One(32);
        let cell = UnitCell::lamellar(10.0).unwrap();

        let monomer = Monomer::new(0, 1.0);
        let point = Point::new(monomer, 0.5);
        let species: Vec<Species> = vec![point.into()];

        // When: System with only point particles
        let mut system = System::new(mesh, cell, species).unwrap();
        system.update();

        // Then: Should compute stress from translational entropy
        let stress = system.stress();
        assert_eq!(stress.len(), 1);

        // Point particle stress is -φ/V (isotropic pressure)
        let volume = system.domain().cell().volume();
        let expected = -0.5 / volume;
        assert_approx_eq!(f64, stress[0], expected, epsilon = 1e-10);
    }
}
