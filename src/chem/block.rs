use super::Monomer;

#[derive(Debug, Clone, Copy)]
pub struct Block {
    pub monomer: Monomer,
    pub repeat_units: usize,
    pub segment_length: f64,
}

impl Block {
    pub fn new(monomer: Monomer, repeat_units: usize, segment_length: f64) -> Self {
        Self {
            monomer,
            repeat_units,
            segment_length,
        }
    }

    pub fn volume(&self) -> f64 {
        self.monomer.volume * (self.repeat_units as f64)
    }
}
