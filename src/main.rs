use fts::{chem::*, domain::*, simulation::*, system::*};
use ndarray_rand::rand_distr::Normal;
use rand::{rngs::SmallRng, SeedableRng};

fn main() {
    let mesh = Mesh::One(128);
    let cell = UnitCell::lamellar(10.0).unwrap();

    let monomer_a = Monomer::new(0, 1.0);
    let monomer_b = Monomer::new(1, 1.0);

    let block_a = Block::new(monomer_a, 50, 1.0);
    let block_b = Block::new(monomer_b, 50, 1.0);

    let polymer = Polymer::new(vec![block_a, block_b], 100, 1.0);
    let species: Vec<Species> = vec![polymer.into()];

    let mut system = System::new(mesh, cell, species).unwrap();
    system.interaction_mut().set_chi(0, 1, 0.25);

    let distr = Normal::new(0.0, 0.1).unwrap();
    let mut rng = SmallRng::seed_from_u64(12345);
    system.sample_fields(&distr, &mut rng);

    // SCFT
    let config = SCFTConfig {
        steps: 1000,
        step_size: 0.25,
        field_tolerance: 1e-5,
    };
    let simulation = SCFT::new(config);
    let result = simulation.run(&mut system);
    dbg!(result);
}
