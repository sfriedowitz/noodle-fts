use itertools::{iproduct, Itertools};
use ndarray::{Array2, Axis};

use super::{cell::UnitCell, mesh::Mesh};
use crate::{
    error::{FTSError, Result},
    fields::RField,
    math::{fftfreq, rfftfreq, TWO_PI},
};

fn get_kvecs_1d(nx: usize) -> Array2<f64> {
    let dx = 1.0 / (TWO_PI * nx as f64);
    let freq_x = rfftfreq(nx, Some(dx));
    iproduct!(freq_x).map(|(kx,)| [kx]).collect_vec().into()
}

fn get_kvecs_2d(nx: usize, ny: usize) -> Array2<f64> {
    let dx = 1.0 / (TWO_PI * nx as f64);
    let dy = 1.0 / (TWO_PI * ny as f64);

    let freq_x = fftfreq(nx, Some(dx));
    let freq_y = rfftfreq(ny, Some(dy));

    iproduct!(freq_x, freq_y)
        .map(|(kx, ky)| [kx, ky])
        .collect_vec()
        .into()
}

fn get_kvecs_3d(nx: usize, ny: usize, nz: usize) -> Array2<f64> {
    let dx = 1.0 / (TWO_PI * nx as f64);
    let dy = 1.0 / (TWO_PI * ny as f64);
    let dz = 1.0 / (TWO_PI * nz as f64);

    let freq_x = fftfreq(nx, Some(dx));
    let freq_y = fftfreq(ny, Some(dy));
    let freq_z = rfftfreq(nz, Some(dz));

    iproduct!(freq_x, freq_y, freq_z)
        .map(|(kx, ky, kz)| [kx, ky, kz])
        .collect_vec()
        .into()
}

#[derive(Debug, Clone)]
pub struct Domain {
    mesh: Mesh,
    cell: UnitCell,
    ksq: RField,
}

impl Domain {
    pub fn new(mesh: Mesh, cell: UnitCell) -> Result<Self> {
        if mesh.ndim() != cell.ndim() {
            return Err(FTSError::DimensionMismatch("mesh".into(), "cell".into()));
        }
        let ksq = RField::zeros(mesh.kmesh());
        Ok(Self { mesh, cell, ksq })
    }

    pub fn ndim(&self) -> usize {
        self.mesh.ndim()
    }

    pub fn mesh(&self) -> Mesh {
        self.mesh
    }

    pub fn cell(&self) -> &UnitCell {
        &self.cell
    }

    pub fn mesh_size(&self) -> usize {
        self.mesh.size()
    }

    pub fn ksq(&self) -> &RField {
        &self.ksq
    }

    pub fn update_ksq(&mut self) -> Result<()> {
        // TODO: Figure out how to do this without allocating new arrays each time
        let kvecs = match self.mesh {
            Mesh::One(nx) => get_kvecs_1d(nx),
            Mesh::Two(nx, ny) => get_kvecs_2d(nx, ny),
            Mesh::Three(nx, ny, nz) => get_kvecs_3d(nx, ny, nz),
        };
        let kvecs_scaled = kvecs.dot(self.cell.metric_inv());
        let ksq = (kvecs * kvecs_scaled)
            .sum_axis(Axis(1))
            .into_shape(self.mesh.kmesh())?;
        self.ksq.assign(&ksq);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::Domain;
    use crate::domain::{Mesh, UnitCell};

    #[test]
    fn test_ksq() {
        let mesh = Mesh::Two(4, 4);
        let cell = UnitCell::square(100.0).unwrap();

        let mut domain = Domain::new(mesh, cell).unwrap();

        let now = Instant::now();
        domain.update_ksq();
        let elapsed = now.elapsed();

        // dbg!(domain.ksq());

        dbg!(elapsed);
    }
}
