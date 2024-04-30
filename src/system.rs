use crate::{
    chem::monomer::Monomer,
    domain::{domain::Domain, RField},
};

pub struct System {
    domain: Domain,
    monomers: Vec<Monomer>,
    omegas: Vec<RField>,
}
