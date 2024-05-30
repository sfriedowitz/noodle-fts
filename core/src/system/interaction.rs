use itertools::iproduct;
use ndarray::Array2;

use crate::fields::{FieldExt, RField};

#[derive(Debug)]
pub struct Interaction {
    chi: Array2<f64>,
}

impl Interaction {
    pub fn new(nmonomer: usize) -> Self {
        let chi = Array2::zeros((nmonomer, nmonomer));
        Self { chi }
    }

    pub fn nmonomer(&self) -> usize {
        self.chi.shape()[0]
    }

    pub fn set_chi(&mut self, i: usize, j: usize, chi: f64) {
        if i != j {
            self.chi[[i, j]] = chi;
            self.chi[[j, i]] = chi;
        }
    }

    pub fn energy(&self, concentrations: &[RField]) -> f64 {
        let mut energy = 0.0;
        for (i, j) in self.iter_pairs() {
            let conc_i = &concentrations[i];
            let conc_j = &concentrations[j];
            let chi_ij = self.chi[[i, j]];
            energy += conc_i.fold_with(conc_j, 0.0, |acc, ci, cj| acc + 0.5 * chi_ij * ci * cj);
        }
        energy / concentrations[0].len() as f64
    }

    pub fn energy_bulk(&self, concentrations: &[f64]) -> f64 {
        let mut energy = 0.0;
        for (i, j) in self.iter_pairs() {
            let conc_i = &concentrations[i];
            let conc_j = &concentrations[j];
            let chi_ij = self.chi[[i, j]];
            energy += 0.5 * chi_ij * conc_i * conc_j;
        }
        energy
    }

    pub fn add_gradients(&self, concentrations: &[RField], fields: &mut [RField]) {
        for (i, j) in self.iter_pairs() {
            let omega_i = &mut fields[i];
            let conc_j = &concentrations[j];
            let chi_ij = self.chi[[i, j]];
            omega_i.zip_mut_with(&conc_j, |wi, cj| *wi += chi_ij * cj);
        }
    }

    fn iter_pairs(&self) -> impl Iterator<Item = (usize, usize)> {
        let nm = self.chi.shape()[0];
        iproduct![0..nm, 0..nm].filter(|(i, j)| i != j)
    }
}

#[cfg(test)]
mod tests {
    use float_cmp::assert_approx_eq;

    use super::*;
    use crate::domain::Mesh;

    #[test]
    fn test_iter_pairs() {
        let nmonomer = 3;
        let itx = Interaction::new(nmonomer);

        let got_pairs: Vec<_> = itx.iter_pairs().collect();
        let mut expected_pairs: Vec<(usize, usize)> = vec![];
        for i in 0..nmonomer {
            for j in 0..nmonomer {
                if i != j {
                    expected_pairs.push((i, j));
                }
            }
        }

        assert_eq!(got_pairs, expected_pairs);
    }

    #[test]
    fn test_energy() {
        let mut itx = Interaction::new(2);
        itx.set_chi(0, 1, 2.0);

        let mesh = Mesh::One(10);
        let conc_0 = RField::from_elem(mesh, 0.75);
        let conc_1 = RField::from_elem(mesh, 0.25);
        let concentrations = vec![conc_0, conc_1];
        let energy = itx.energy(&concentrations);

        // 2 * 0.75 * 0.25 = 0.375
        assert_approx_eq!(f64, energy, 0.375)
    }

    #[test]
    fn test_energy_bulk() {
        let mut itx = Interaction::new(2);
        itx.set_chi(0, 1, 2.0);

        let concentrations = vec![0.9, 0.1];
        let energy = itx.energy_bulk(&concentrations);

        // 2 * 0.9 * 0.1 = 0.18
        assert_approx_eq!(f64, energy, 0.18)
    }
}
