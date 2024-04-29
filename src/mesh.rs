#[derive(Clone, Debug)]
pub struct Mesh {
    dimensions: Vec<usize>,
}

impl Mesh {
    pub fn new(dimensions: Vec<usize>) -> Self {
        Self {
            dimensions: dimensions.into(),
        }
    }

    pub fn complex(&self) -> Self {
        let mut k_dimensions = self.dimensions.clone();
        let last_dim = k_dimensions.last_mut().unwrap();
        *last_dim = *last_dim / 2 + 1;
        Self::new(k_dimensions)
    }

    pub fn n_dim(&self) -> usize {
        self.dimensions.len()
    }

    pub fn dimensions(&self) -> &[usize] {
        &self.dimensions
    }

    pub fn size(&self) -> usize {
        self.dimensions.iter().product()
    }
}
