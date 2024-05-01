use super::block::Block;

#[derive(Debug, Clone)]
pub struct Polymer {
    pub blocks: Vec<Block>,
}

impl Polymer {
    pub fn builder() -> PolymerBuilder {
        PolymerBuilder::default()
    }
}

#[derive(Default)]
pub struct PolymerBuilder {
    blocks: Vec<Block>,
}

impl PolymerBuilder {
    pub fn with_block(mut self, monomer_id: usize, length: u32, kuhn: f64) -> Self {
        let block = Block::new(self.blocks.len(), monomer_id, length, kuhn);
        self.blocks.push(block);
        self
    }

    pub fn build(self) -> Polymer {
        Polymer {
            blocks: self.blocks,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Polymer;

    #[test]
    fn test_builder() {
        let polymer = Polymer::builder()
            .with_block(0, 100, 1.0)
            .with_block(1, 100, 1.0)
            .build();

        assert!(polymer.blocks.len() == 2);
        assert!(polymer.blocks[0].id == 0);
        assert!(polymer.blocks[1].id == 1);
    }
}
