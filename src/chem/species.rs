use std::collections::HashSet;

use enum_dispatch::enum_dispatch;

use super::{block::Block, monomer::Monomer};
use crate::error::{FTSError, Result};

#[enum_dispatch]
pub trait SpeciesDescription {
    fn fraction(&self) -> f64;

    fn monomer_fraction(&self, id: usize) -> f64;

    fn size(&self, monomers: &[Monomer]) -> f64;
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

#[derive(Debug, Clone)]
pub struct Point {
    fraction: f64,
    monomer_id: usize,
}

impl Point {
    pub fn new(monomer_id: usize, fraction: f64) -> Self {
        Self {
            monomer_id,
            fraction,
        }
    }

    pub fn monomer_id(&self) -> usize {
        self.monomer_id
    }
}

impl SpeciesDescription for Point {
    fn fraction(&self) -> f64 {
        self.fraction
    }

    fn monomer_fraction(&self, id: usize) -> f64 {
        if id == self.monomer_id {
            1.0
        } else {
            0.0
        }
    }

    fn size(&self, monomers: &[Monomer]) -> f64 {
        monomers[self.monomer_id].size
    }
}

#[derive(Debug, Clone)]
pub struct Polymer {
    fraction: f64,
    contour_steps: usize,
    blocks: Vec<Block>,
}

impl Polymer {
    pub fn new(fraction: f64, contour_steps: usize, blocks: Vec<Block>) -> Result<Self> {
        match blocks.len() {
            0 => Err(FTSError::Validation(
                "Polymer must contain at least one block".into(),
            )),
            _ => Ok(Self {
                fraction,
                contour_steps,
                blocks,
            }),
        }
    }

    pub fn nblock(&self) -> usize {
        return self.blocks.len();
    }

    pub fn with_block(mut self, block: Block) -> Self {
        self.blocks.push(block);
        self
    }

    pub fn monomer_ids(&self) -> HashSet<usize> {
        self.blocks.iter().map(|b| b.monomer_id).collect()
    }
}

impl SpeciesDescription for Polymer {
    fn fraction(&self) -> f64 {
        self.fraction
    }

    fn monomer_fraction(&self, id: usize) -> f64 {
        let total: f64 = self
            .blocks
            .iter()
            .map(|b| if b.monomer_id == id { 1.0 } else { 0.0 })
            .sum();
        total / self.nblock() as f64
    }

    fn size(&self, monomers: &[Monomer]) -> f64 {
        self.blocks
            .iter()
            .map(|b| monomers[b.monomer_id].size * (b.repeat_units as f64))
            .sum()
    }
}

#[cfg(test)]
mod tests {

    use crate::chem::{Block, Monomer, Point, Polymer, Species, SpeciesDescription};

    #[test]
    fn test_point_species() {
        let monomer = Monomer::new(0, 1.0);
        let species: Species = Point::new(monomer.id, 1.0).into();

        assert!(species.monomer_fraction(monomer.id) == 1.0);
        assert!(species.monomer_fraction(1) == 0.0);

        assert!(species.size(&[monomer]) == monomer.size);
    }

    #[test]
    fn test_polymer_species() {
        let monomer_a = Monomer::new(0, 1.0);
        let monomer_b = Monomer::new(1, 1.0);

        let block_a = Block::new(monomer_a.id, 100, 1.0);
        let block_b = Block::new(monomer_b.id, 100, 1.0);

        let species: Species = Polymer::new(1.0, 100, vec![block_a, block_b])
            .unwrap()
            .into();

        assert!(species.monomer_fraction(monomer_a.id) == 0.5);
        assert!(species.monomer_fraction(monomer_b.id) == 0.5);

        assert!(species.size(&[monomer_a, monomer_b]) == 200.0);
    }
}
