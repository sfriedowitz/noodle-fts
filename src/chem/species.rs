use enum_dispatch::enum_dispatch;
use itertools::Itertools;

use super::{block::Block, monomer::Monomer};

#[enum_dispatch]
pub trait SpeciesDescription {
    fn size(&self) -> f64;

    fn fraction(&self) -> f64;

    fn monomers(&self) -> Vec<Monomer>;

    fn monomer_fraction(&self, id: usize) -> f64;
}

/// Enumeration of molecular species types.
///
/// Species implement the `SpeciesDescription` trait,
/// which provides the basic interface for computing system-level species quantities.
#[enum_dispatch(SpeciesDescription)]
#[derive(Debug, Clone)]
pub enum Species {
    Point,
    Polymer,
}

#[derive(Debug, Clone, Copy)]
pub struct Point {
    pub monomer: Monomer,
    pub fraction: f64,
}

impl Point {
    pub fn new(monomer: Monomer, fraction: f64) -> Self {
        Self { monomer, fraction }
    }
}

impl SpeciesDescription for Point {
    fn size(&self) -> f64 {
        self.monomer.size
    }

    fn fraction(&self) -> f64 {
        self.fraction
    }

    fn monomers(&self) -> Vec<Monomer> {
        Vec::from([self.monomer])
    }

    fn monomer_fraction(&self, id: usize) -> f64 {
        if id == self.monomer.id {
            1.0
        } else {
            0.0
        }
    }
}

#[derive(Debug, Clone)]
pub struct Polymer {
    pub blocks: Vec<Block>,
    pub contour_steps: usize,
    pub fraction: f64,
}

impl Polymer {
    pub fn new(blocks: Vec<Block>, contour_steps: usize, fraction: f64) -> Self {
        Self {
            blocks,
            contour_steps,
            fraction,
        }
    }

    pub fn nblock(&self) -> usize {
        return self.blocks.len();
    }
}

impl SpeciesDescription for Polymer {
    fn size(&self) -> f64 {
        self.blocks
            .iter()
            .map(|b| b.monomer.size * (b.repeat_units as f64))
            .sum()
    }

    fn fraction(&self) -> f64 {
        self.fraction
    }

    fn monomers(&self) -> Vec<Monomer> {
        self.blocks
            .iter()
            .map(|b| b.monomer)
            .unique_by(|monomer| monomer.id)
            .collect()
    }

    fn monomer_fraction(&self, id: usize) -> f64 {
        let total: f64 = self
            .blocks
            .iter()
            .map(|b| if b.monomer.id == id { 1.0 } else { 0.0 })
            .sum();
        total / self.nblock() as f64
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_point_species() {
        let monomer = Monomer::new(0, 1.0);
        let species: Species = Point::new(monomer, 1.0).into();

        assert!(species.monomer_fraction(monomer.id) == 1.0);
        assert!(species.monomer_fraction(1) == 0.0);

        assert!(species.size() == monomer.size);
    }

    #[test]
    fn test_polymer_species() {
        let monomer_a = Monomer::new(0, 1.0);
        let monomer_b = Monomer::new(1, 1.0);

        let block_a = Block::new(monomer_a, 100, 1.0);
        let block_b = Block::new(monomer_b, 100, 1.0);

        let species: Species = Polymer::new(vec![block_a, block_b], 100, 1.0).into();

        assert!(species.monomer_fraction(monomer_a.id) == 0.5);
        assert!(species.monomer_fraction(monomer_b.id) == 0.5);

        assert!(species.size() == 200.0);
    }
}
