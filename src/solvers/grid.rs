use std::ops::Range;

use crate::chem::{Block, Monomer, Polymer, SpeciesDescription};

#[derive(Debug, Clone, Copy)]
pub struct GridBlock {
    pub monomer_id: usize,
    pub monomer_size: f64,
    pub segment_length: f64,
    pub start: usize,
    pub end: usize,
    pub ds: f64,
}

impl GridBlock {
    pub fn new(block: Block, start: usize, end: usize, ds: f64) -> Self {
        Self {
            monomer_id: block.monomer.id,
            monomer_size: block.monomer.size,
            segment_length: block.segment_length,
            start,
            end,
            ds,
        }
    }
}

#[derive(Debug)]
pub struct ContourGrid {
    ns: usize,
    blocks: Vec<GridBlock>,
}

impl ContourGrid {
    pub fn new(polymer: &Polymer) -> Self {
        let target_ds = polymer.size() / polymer.contour_steps as f64;
        let mut grid_blocks: Vec<GridBlock> = vec![];

        let mut ns_total = 0;
        for b in polymer.blocks.iter().cloned() {
            // ns is guaranteed to be at least 3 for each block,
            // which results in an even number of contour steps per block
            let ns = (b.size() / (2.0 * target_ds) + 0.5).floor() as usize;
            let ns = ns.max(1);
            let ns = 2 * ns + 1;

            // actual_ns counter runs 1 head of the block start positions
            let start = ns_total;
            let end = start + ns;
            ns_total += ns;

            let ds = b.size() / ((ns - 1) as f64);

            grid_blocks.push(GridBlock::new(b, start, end, ds))
        }

        Self {
            ns: ns_total,
            blocks: grid_blocks,
        }
    }

    pub fn ns(&self) -> usize {
        self.ns
    }

    pub fn nblock(&self) -> usize {
        self.blocks.len()
    }

    pub fn get_block(&self, s: usize) -> Option<GridBlock> {
        self.blocks
            .iter()
            .copied()
            .find(|b| s >= b.start && s < b.end)
    }
}

#[derive(Debug)]
pub struct GridIterator<'a> {
    index: usize,
    grid: &'a ContourGrid,
}

impl<'a> Iterator for GridIterator<'a> {
    type Item = GridBlock;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.grid.nblock() {
            return None;
        }
        let item = self.grid.blocks[self.index];
        self.index += 1;
        Some(item)
    }
}

impl<'a> DoubleEndedIterator for GridIterator<'a> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.index == 0 {
            return None;
        }
        self.index -= 1;
        let item = self.grid.blocks[self.index];
        Some(item)
    }
}

impl<'a> IntoIterator for &'a ContourGrid {
    type Item = GridBlock;

    type IntoIter = GridIterator<'a>;

    fn into_iter(self) -> Self::IntoIter {
        GridIterator {
            index: 0,
            grid: self,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ContourGrid;
    use crate::chem::{Block, Monomer, Polymer};

    #[test]
    fn test_grid_discretization() {
        let monomer_a = Monomer::new(0, 1.0);
        let monomer_b = Monomer::new(1, 1.0);
        let monomer_c = Monomer::new(2, 1.0);

        let block_a = Block::new(monomer_a, 100, 1.0);
        let block_b = Block::new(monomer_b, 100, 1.0);
        let block_c = Block::new(monomer_c, 100, 1.0);

        let polymer = Polymer::new(vec![block_a, block_b, block_c], 300, 1.0);

        let x: f64 = (0..300).map(|_| 1.0).sum();
        dbg!(x);

        let grid = ContourGrid::new(&polymer);

        dbg!(grid);
    }
}
