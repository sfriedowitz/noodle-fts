#[derive(Debug, Clone, Copy)]
pub struct Block {
    pub id: usize,
    pub monomer_id: usize,
    pub length: u32,
    pub kuhn: f64,
}

impl Block {
    pub fn new(id: usize, monomer_id: usize, length: u32, kuhn: f64) -> Self {
        Block {
            id,
            monomer_id,
            length,
            kuhn,
        }
    }
}
