use crate::{chem::Monomer, field::RField};

pub struct SolverInput<'a> {
    monomers: &'a [Monomer],
    fields: &'a [RField],
    ksq: &'a RField,
}

pub trait SpeciesSolver {
    fn solve<'a>(&mut self, input: SolverInput<'a>);
}
