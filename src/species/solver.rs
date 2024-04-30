use super::ensemble::Ensemble;
use crate::{
    chem::monomer::Monomer,
    domain::{domain::Domain, RField},
};

pub struct SolverInput<'a> {
    pub ensemble: Ensemble,
    pub domain: &'a Domain,
    pub omegas: &'a Vec<RField>,
    pub monomers: &'a Vec<Monomer>,
}

pub trait SpeciesSolver {
    fn solve<'a>(&mut self, input: SolverInput<'a>);
}
