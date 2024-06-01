use fts::system::{Interaction, System};

use crate::impl_conversions;

pub struct PyInteraction {
    core: Interaction,
}

impl_conversions!(Interaction, PyInteraction);

pub struct PySystem {
    core: System,
}

impl_conversions!(System, PySystem);
