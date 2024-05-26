use itertools::iproduct;
use ndarray::{Array2, Zip};

use crate::RField;

pub struct Interaction {
    chi: Array2<f64>,
}

impl Interaction {
    pub fn new(nmonomer: usize) -> Self {
        let chi = Array2::zeros((nmonomer, nmonomer));
        Self { chi }
    }

    pub fn set_chi(&mut self, i: usize, j: usize, chi: f64) {
        if i != j {
            self.chi[[i, j]] = chi;
            self.chi[[j, i]] = chi;
        }
    }

    pub fn energy(&self, density: &[RField]) -> f64 {
        let mut energy = 0.0;
        for (i, j) in self.iter_pairs() {
            let rho_i = &density[i];
            let rho_j = &density[j];
            let chi_ij = self.chi[[i, j]];
            Zip::from(rho_i)
                .and(rho_j)
                .for_each(|ri, rj| energy += 0.5 * chi_ij * ri * rj);
        }
        energy / density[0].len() as f64
    }

    pub fn energy_bulk(&self, density: &[f64]) -> f64 {
        let mut energy = 0.0;
        for (i, j) in self.iter_pairs() {
            let rho_i = &density[i];
            let rho_j = &density[j];
            let chi_ij = self.chi[[i, j]];
            energy += 0.5 * chi_ij * rho_i * rho_j;
        }
        energy
    }

    pub fn add_potentials(&self, density: &[RField], fields: &mut [RField]) {
        for omega in fields.iter_mut() {
            omega.fill(0.0);
        }
        for (i, j) in self.iter_pairs() {
            let omega_i = &mut fields[i];
            let rho_j = &density[j];
            let chi_ij = self.chi[[i, j]];
            Zip::from(omega_i)
                .and(rho_j)
                .for_each(|wi, rj| *wi += chi_ij * rj);
        }
    }

    fn iter_pairs(&self) -> impl Iterator<Item = (usize, usize)> {
        let nm = self.chi.shape()[0];
        iproduct![0..nm, 0..nm].filter(|(i, j)| i != j)
    }
}

#[cfg(test)]
mod tests {
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
        let rho_0 = RField::from_elem(mesh, 0.75);
        let rho_1 = RField::from_elem(mesh, 0.25);
        let density = vec![rho_0, rho_1];
        let energy = itx.energy(&density);

        // 2 * 0.75 * 0.25 = 0.375
        assert_eq!(energy, 0.375)
    }

    #[test]
    fn test_energy_bulk() {
        let mut itx = Interaction::new(2);
        itx.set_chi(0, 1, 2.0);

        let density = vec![0.5, 0.5];
        let energy = itx.energy_bulk(&density);

        // 2 * 0.5 * 0.5 = 0.5
        assert_eq!(energy, 0.5)
    }
}
