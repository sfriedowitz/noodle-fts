use super::monomer::Monomer;

#[derive(Debug, Clone, Copy)]
pub struct Block {
    pub monomer_id: usize,
    pub repeat_units: usize,
    pub segment_length: f64,
}

impl Block {
    pub fn new(monomer_id: usize, repeat_units: usize, segment_length: f64) -> Self {
        Self {
            monomer_id,
            repeat_units,
            segment_length,
        }
    }
}
