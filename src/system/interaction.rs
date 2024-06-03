use std::collections::HashMap;

use crate::fields::{FieldOps, RField};

#[derive(Debug)]
pub struct Interaction {
    pairs: HashMap<(usize, usize), f64>,
}

impl Interaction {
    pub fn new() -> Self {
        Self {
            pairs: HashMap::new(),
        }
    }

    pub fn set_chi(&mut self, i: usize, j: usize, chi: f64) {
        if i != j {
            self.pairs.insert((i, j), chi);
            self.pairs.insert((j, i), chi);
        }
    }

    pub fn energy(&self, concentrations: &HashMap<usize, RField>) -> f64 {
        let mut energy = 0.0;
        for ((i, j), chi_ij) in self.pairs.iter() {
            if let Some(conc_i) = concentrations.get(i) {
                if let Some(conc_j) = concentrations.get(j) {
                    let prod = conc_i.fold_with(conc_j, 0.0, |acc, ci, cj| acc + 0.5 * chi_ij * ci * cj);
                    energy += prod / conc_i.len() as f64
                }
            }
        }
        energy
    }

    pub fn energy_bulk(&self, concentrations: &HashMap<usize, f64>) -> f64 {
        let mut energy = 0.0;
        for ((i, j), chi_ij) in self.pairs.iter() {
            if let Some(conc_i) = concentrations.get(i) {
                if let Some(conc_j) = concentrations.get(j) {
                    energy += 0.5 * chi_ij * conc_i * conc_j;
                }
            }
        }
        energy
    }

    pub fn add_gradients(
        &self,
        concentrations: &HashMap<usize, RField>,
        fields: &mut HashMap<usize, RField>,
    ) {
        for ((i, j), chi_ij) in self.pairs.iter() {
            if let Some(omega_i) = fields.get_mut(i) {
                if let Some(conc_j) = concentrations.get(j) {
                    omega_i.zip_mut_with(&conc_j, |wi, cj| *wi += chi_ij * cj);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use float_cmp::assert_approx_eq;

    use super::*;
    use crate::domain::Mesh;

    #[test]
    fn test_energy() {
        let mut itx = Interaction::new();
        itx.set_chi(0, 1, 2.0);

        let mesh = Mesh::One(10);
        let conc_0 = RField::from_elem(mesh, 0.75);
        let conc_1 = RField::from_elem(mesh, 0.25);
        let concentrations = [(0, conc_0), (1, conc_1)].into();
        let energy = itx.energy(&concentrations);

        // 2 * 0.75 * 0.25 = 0.375
        assert_approx_eq!(f64, energy, 0.375)
    }

    #[test]
    fn test_energy_bulk() {
        let mut itx = Interaction::new();
        itx.set_chi(0, 1, 2.0);

        let concentrations = [(0, 0.9), (1, 0.1), (2, -0.5)].into();
        let energy = itx.energy_bulk(&concentrations);

        // 2 * 0.9 * 0.1 = 0.18
        assert_approx_eq!(f64, energy, 0.18)
    }
}
