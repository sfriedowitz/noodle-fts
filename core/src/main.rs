use std::time::Instant;

use fts::{chem::*, domain::*, system::*, updaters::*};
use ndarray_rand::rand_distr::Normal;

fn main() {
    let mesh = Mesh::Two(128, 128);
    let cell = UnitCell::square(10.0).unwrap();
    let domain = Domain::new(mesh, cell).unwrap();

    let monomer_a = Monomer::new(0, 1.0);
    let monomer_b = Monomer::new(1, 1.0);

    let block_a = Block::new(monomer_a, 50, 1.0);
    let block_b = Block::new(monomer_b, 50, 1.0);

    let polymer = Polymer::new(vec![block_a, block_b], 200, 1.0);
    let species: Vec<Species> = vec![polymer.into()];

    let mut itx = Interaction::new(2);
    itx.set_chi(0, 1, 0.2);

    let mut system = System::new(domain, itx, species).unwrap();

    let distr = Normal::new(0.0, 0.1).unwrap();
    let mut rng = rand::thread_rng();
    system.sample_fields(&distr, &mut rng);
    system.update();

    let mut updater = EulerUpdater::new(&system, 0.01);

    for step in 0..1000 {
        let now = Instant::now();
        updater.step(&mut system);
        let elapsed = now.elapsed().as_secs_f64();
        let f = system.free_energy();
        if f.is_nan() {
            break;
        }
        println!("Step = {}, f = {}, time = {}", step, f, elapsed);
    }
    dbg!(system.free_energy_bulk());
    dbg!(system.total_concentration());
}
