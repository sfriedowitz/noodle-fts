use std::rc::Rc;

use crate::{domain::Mesh, fields::RField};

#[derive(Debug, Clone, Copy)]
pub enum PropagatorDirection {
    Forward,
    Reverse,
}

#[derive(Debug)]
pub struct BlockPropagator {
    qfields: Vec<RField>,
    source: Option<Rc<BlockPropagator>>,
}

impl BlockPropagator {
    pub fn new(mesh: Mesh, ngrid: usize) -> Self {
        let qfields = (0..ngrid).map(|_| RField::zeros(mesh)).collect();
        Self {
            qfields,
            source: None,
        }
    }

    pub fn len(&self) -> usize {
        self.qfields.len()
    }

    pub fn qfields(&self) -> &[RField] {
        &self.qfields
    }

    pub fn qfields_mut(&mut self) -> &mut [RField] {
        &mut self.qfields
    }

    pub fn set_source(&mut self, source: Rc<BlockPropagator>) {
        self.source = Some(source)
    }
}
