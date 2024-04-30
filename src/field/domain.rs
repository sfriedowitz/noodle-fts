use super::{cell::UnitCell, mesh::Mesh};

pub struct Domain {
    mesh: Mesh,
    cell: UnitCell,
}
