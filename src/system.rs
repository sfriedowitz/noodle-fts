use std::collections::HashMap;

use crate::field::{domain::Domain, RField};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ensemble {
    Closed,
    Open,
}

pub struct System {
    domain: Domain,
    fields: HashMap<usize, RField>,
}
