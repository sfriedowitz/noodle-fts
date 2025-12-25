#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Monomer {
    pub id: usize,
    pub volume: f64,
}

impl Monomer {
    pub fn new(id: usize, volume: f64) -> Self {
        Self { id, volume }
    }
}
