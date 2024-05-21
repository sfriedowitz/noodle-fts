use super::{cell::UnitCell, mesh::Mesh};
use crate::prelude::RField;

#[derive(Debug, Clone)]
pub struct Domain {
    mesh: Mesh,
    cell: UnitCell,
    ksq: RField,
}

impl Domain {
    pub fn new(mesh: Mesh, cell: UnitCell) -> Self {
        let ksq = RField::zeros(mesh.kmesh());
        Self { mesh, cell, ksq }
    }

    pub fn mesh_size(&self) -> usize {
        self.mesh.size()
    }

    pub fn update_ksq(&mut self) {
        todo!()
    }
}
