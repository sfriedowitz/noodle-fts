#[derive(Clone, Debug)]
pub struct Mesh {
    dimensions: Vec<usize>,
    k_dimensions: Vec<usize>,
}

impl Mesh {
    pub fn new(dimensions: &[usize]) -> Self {
        let mut k_dimensions = dimensions.to_vec();
        let last_dim = k_dimensions.last_mut().unwrap();
        *last_dim = *last_dim / 2 + 1;

        Self {
            dimensions: dimensions.to_vec(),
            k_dimensions,
        }
    }

    pub fn n_dim(&self) -> usize {
        self.dimensions.len()
    }

    pub fn dimensions(&self) -> &[usize] {
        &self.dimensions
    }

    pub fn k_dimensions(&self) -> &[usize] {
        &self.k_dimensions
    }
}
