use std::collections::HashMap;

use crate::{mesh::Mesh, types::RField};

struct System {
    mesh: Mesh,
    fields: HashMap<usize, RField>,
}
