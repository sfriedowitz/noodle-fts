use std::collections::HashMap;

use enum_dispatch::enum_dispatch;

use super::{PointSolver, PolymerSolver};
use crate::{
    chem::Species,
    domain::{Domain, Mesh},
    fields::RField,
};

#[enum_dispatch]
pub trait SolverOps {
    fn species(&self) -> Species;

    fn partition(&self) -> f64;

    fn concentrations(&self) -> &HashMap<usize, RField>;

    fn solve(&mut self, fields: &HashMap<usize, RField>, ksq: &RField);

    /// Compute the stress tensor contribution from this species.
    ///
    /// Returns stress components as a flattened vector:
    /// - 1D: [σ_xx]
    /// - 2D: [σ_xx, σ_yy, σ_xy]
    /// - 3D: [σ_xx, σ_yy, σ_zz, σ_xy, σ_xz, σ_yz]
    fn stress(&mut self, domain: &Domain) -> Vec<f64>;
}

#[enum_dispatch(SolverOps)]
#[derive(Debug)]
pub enum SpeciesSolver {
    PointSolver,
    PolymerSolver,
}

impl SpeciesSolver {
    pub fn new(mesh: Mesh, species: impl Into<Species>) -> Self {
        match species.into() {
            Species::Point(point) => PointSolver::new(mesh, point).into(),
            Species::Polymer(polymer) => PolymerSolver::new(mesh, polymer).into(),
        }
    }
}
