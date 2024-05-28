use fts::{chem::*, domain::*, system::*, updaters::*};
use ndarray_rand::rand_distr::Normal;

fn main() {
    let mesh = Mesh::One(32);
    let cell = UnitCell::lamellar(10.0).unwrap();
    let domain = Domain::new(mesh, cell).unwrap();

    let monomer_a = Monomer::new(0, 1.0);
    let monomer_b = Monomer::new(1, 1.0);

    let block_a = Block::new(monomer_a, 50, 1.0);
    let block_b = Block::new(monomer_b, 50, 1.0);

    let polymer = Polymer::new(vec![block_a, block_b], 100, 1.0);
    let species: Vec<Species> = vec![polymer.into()];

    let mut itx = Interaction::new(2);
    itx.set_chi(0, 1, 0.12);

    let mut system = System::new(domain, itx, species).unwrap();
    let mut updater = EulerUpdater::new(&system, 0.1);

    let distr = Normal::new(0.0, 0.1).unwrap();
    let mut rng = rand::thread_rng();
    system.sample_fields(&distr, &mut rng);

    for step in 0..250 {
        updater.step(&mut system).unwrap();
        println!("Step = {}, f = {}", step, system.free_energy());
    }

    dbg!(system.free_energy_bulk());
}
