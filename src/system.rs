use crate::{
    chem::monomer::Monomer,
    domain::{domain::Domain, RField},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ensemble {
    Open,
    Closed,
}

pub struct System {
    domain: Domain,
    monomers: Vec<Monomer>,
    species: Vec<Species>,
    fields: Vec<RField>,
}

impl System {
    pub fn nmonomer(&self) -> usize {
        todo!()
    }

    pub fn nspecies(&self) -> usize {
        todo!()
    }
}
