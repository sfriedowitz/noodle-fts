use crate::{impl_py_conversions, simulation::FieldUpdater, system::System};

pub struct PySystem(System);

impl_py_conversions!(System, PySystem);

pub struct PyFieldUpdater(FieldUpdater);

impl_py_conversions!(FieldUpdater, PyFieldUpdater);
