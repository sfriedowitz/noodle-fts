use super::{fft::FFT, mesh::Mesh, RField};

pub struct Domain {
    mesh: Mesh,
    fft: FFT,
    ksq: RField,
}

impl Domain {
    pub fn mesh_size(&self) -> usize {
        self.mesh.size()
    }

    pub fn update_ksq(&mut self) {
        todo!()
    }
}
