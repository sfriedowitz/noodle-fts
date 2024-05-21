#[derive(Debug, Clone, Copy)]
pub struct Monomer {
    pub id: usize,
    pub size: f64,
}

impl Monomer {
    pub fn new(id: usize, size: f64) -> Self {
        Self { id, size }
    }
}
