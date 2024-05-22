use itertools::iproduct;
use ndarray::{Array1, Array2, Array3};

use super::{cell::UnitCell, mesh::Mesh};
use crate::{
    error::{FTSError, Result},
    fields::RField,
    math::{fftshift_index, TWO_PI},
};

#[derive(Debug, Clone)]
pub struct Domain {
    mesh: Mesh,
    cell: UnitCell,
}

impl Domain {
    pub fn new(mesh: Mesh, cell: UnitCell) -> Result<Self> {
        if mesh.ndim() != cell.ndim() {
            Err(FTSError::DimensionMismatch(mesh.ndim(), cell.ndim()))
        } else {
            Ok(Self { mesh, cell })
        }
    }

    pub fn ndim(&self) -> usize {
        self.mesh.ndim()
    }

    pub fn mesh_size(&self) -> usize {
        self.mesh.size()
    }

    pub fn ksq(&self) -> RField {
        let metric_inv = self.cell.metric_inv();
        match self.mesh.kmesh() {
            Mesh::One(nx_half) => {
                let mut kvec = Array1::zeros(1);
                Array1::from_shape_fn((nx_half), |ix| {
                    let jx = ix as f64;
                    kvec[0] = TWO_PI * jx;
                    kvec.dot(&metric_inv.dot(&kvec))
                })
                .into_dyn()
            }
            Mesh::Two(nx, ny_half) => {
                let mut kvec = Array1::zeros(2);
                Array2::from_shape_fn((nx, ny_half), |(ix, iy)| {
                    let jx = fftshift_index(ix, nx);
                    let jy = iy as f64;
                    kvec[0] = TWO_PI * (ix as f64);
                    kvec[1] = TWO_PI * (iy as f64);
                    kvec.dot(&metric_inv.dot(&kvec))
                })
                .into_dyn()
            }
            Mesh::Three(nx, ny, nz_half) => {
                let mut kvec = Array1::zeros(3);
                Array3::from_shape_fn((nx, ny, nz_half), |(ix, iy, iz)| {
                    let jx = fftshift_index(ix, nx);
                    let jy = fftshift_index(iy, ny);
                    let jz = iz as f64;
                    kvec[0] = TWO_PI * (ix as f64);
                    kvec[1] = TWO_PI * (iy as f64);
                    kvec[2] = TWO_PI * (iz as f64);
                    kvec.dot(&metric_inv.dot(&kvec))
                })
                .into_dyn()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::Domain;
    use crate::domain::{CellParameters, Mesh, UnitCell};

    #[test]
    fn test_ksq() {
        let mesh = Mesh::Two(128, 128);
        let cell = UnitCell::new(CellParameters::Hexagonal(10.0)).unwrap();

        let domain = Domain::new(mesh, cell).unwrap();

        let now = Instant::now();
        let _ = domain.ksq();
        let elapsed = now.elapsed();

        // dbg!(domain);
        dbg!(elapsed);
    }
}
