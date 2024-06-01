use fts::{chem::*, domain::*, simulation::*, system::*};
use ndarray_rand::rand_distr::Normal;

fn main() {
    let mesh = Mesh::One(256);
    let cell = UnitCell::lamellar(10.0).unwrap();
    let domain = Domain::new(mesh, cell).unwrap();

    let monomer_a = Monomer::new(0, 1.0);
    let monomer_b = Monomer::new(1, 1.0);

    let block_a = Block::new(monomer_a, 34, 1.0);
    let block_b = Block::new(monomer_b, 66, 1.0);

    let polymer = Polymer::new(vec![block_a, block_b], 50, 1.0);
    let species: Vec<Species> = vec![polymer.into()];

    let mut itx = Interaction::new(2);
    itx.set_chi(monomer_a.id, monomer_b.id, 0.4);

    let mut system = System::new(domain, itx, species).unwrap();

    let distr = Normal::new(0.0, 0.1).unwrap();
    let mut rng = rand::thread_rng();
    system.sample_fields(&distr, &mut rng);
    system.update();

    // SCFT
    let parameters = SCFTParameters {
        nstep: 1000,
        dt: 0.25,
        etol: 1e-5,
    };
    let simulation = SCFT::new(parameters);
    let result = simulation.run(&mut system);
    dbg!(result);
}
