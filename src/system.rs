use std::collections::HashMap;

use crate::{mesh::Mesh, species::Species, types::RField};

struct System {
    mesh: Mesh,
    species: Vec<Species>,
    fields: HashMap<usize, RField>,
}
