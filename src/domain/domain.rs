use super::{cell::UnitCell, fft::FFT, mesh::Mesh};

pub struct Domain {
    mesh: Mesh,
    cell: UnitCell,
    fft: FFT,
}

impl Domain {
    pub fn mesh_size(&self) -> usize {
        self.mesh.size()
    }
}
