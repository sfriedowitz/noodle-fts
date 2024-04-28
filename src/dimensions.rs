pub struct Dimensions {
    data: Vec<usize>,
}

impl Dimensions {
    pub fn new(data: &[usize]) -> Self {
        Self {
            data: data.to_vec(),
        }
    }
}
