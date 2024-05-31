use std::time::Instant;

use fts::{chem::*, domain::*, system::*};
use ndarray_rand::rand_distr::Normal;

fn main() {
    let mesh = Mesh::Two(64, 64);
    let cell = UnitCell::square(10.0).unwrap();
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

    let mut updater = FieldUpdater::new(&system, 0.25, Some(1e-5));

    let total = Instant::now();
    for step in 0..1000 {
        let now = Instant::now();
        updater.step(&mut system);
        let elapsed = now.elapsed().as_secs_f64();
        let f = system.free_energy();
        let err = system.field_error();
        println!(
            "Step = {}, f = {:.5}, err = {:.5e}, time = {:.5e}",
            step, f, err, elapsed
        );
        if updater.is_converged(&system) {
            break;
        }
        if f.is_nan() {
            break;
        }
    }
    println!("Total time = {:.5} sec", total.elapsed().as_secs_f64());
    // dbg!(system.total_concentration());
}
