#[derive(Debug, Clone, Copy)]
pub struct Monomer {
    pub id: usize,
    pub vol: f64,
    pub charge: f64,
}

impl Monomer {
    pub fn new(id: usize, vol: f64, charge: f64) -> Self {
        Self { id, vol, charge }
    }
}
