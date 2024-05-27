mod cell;
mod domain;
mod fft;
mod mesh;

pub use cell::{CellParameters, UnitCell};
pub use domain::Domain;
pub use fft::FFT;
pub use mesh::Mesh;

#[derive(thiserror::Error, Debug)]
pub enum DomainError {
    #[error("mesh and cell dimensions do not match")]
    DimensionMismatch,
}
