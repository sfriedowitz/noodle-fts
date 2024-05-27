use ndarray::{IntoDimension, IxDyn};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mesh {
    One(usize),
    Two(usize, usize),
    Three(usize, usize, usize),
}

impl Mesh {
    pub fn ndim(&self) -> usize {
        match self {
            Self::One(_) => 1,
            Self::Two(_, _) => 2,
            Self::Three(_, _, _) => 3,
        }
    }

    pub fn dimensions(&self) -> Vec<usize> {
        match self {
            Self::One(nx) => vec![*nx],
            Self::Two(nx, ny) => vec![*nx, *ny],
            Self::Three(nx, ny, nz) => vec![*nx, *ny, *nz],
        }
    }

    pub fn kmesh(&self) -> Self {
        match self {
            Self::One(nx) => Self::One(nx / 2 + 1),
            Self::Two(nx, ny) => Self::Two(*nx, ny / 2 + 1),
            Self::Three(nx, ny, nz) => Self::Three(*nx, *ny, nz / 2 + 1),
        }
    }

    pub fn size(&self) -> usize {
        self.dimensions().iter().product()
    }
}

// Allows the `Mesh` type to be used as dimensions for ndarray creation
impl IntoDimension for Mesh {
    type Dim = IxDyn;

    fn into_dimension(self) -> Self::Dim {
        self.dimensions().into_dimension()
    }
}
