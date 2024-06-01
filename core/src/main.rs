use fts::{chem::*, domain::*, simulation::*, system::*};
use ndarray_rand::rand_distr::Normal;

// fn run_fftw() {
//     let dims = [32, 32, 32];
//     let n = 32 * 32 * 32;
//     let nd = 32 * 32 * 17;
//     let mut plan = R2CPlan64::aligned(&dims, Flag::ESTIMATE).unwrap();

//     let mut a = vec![1.0f64; n];
//     let mut b = vec![Complex64::zero(); nd];

//     let now = Instant::now();
//     for _ in 0..1000 {
//         plan.r2c(&mut a, &mut b).unwrap();
//     }
//     let elapsed = now.elapsed();
//     dbg!(elapsed);
// }

// fn run_ndarray() {
//     let mesh = Mesh::Three(32, 32, 32);
//     let mut fft = FFT::new(mesh);

//     let a = RField::zeros(mesh);
//     let mut b = CField::zeros(mesh.kmesh());

//     let now = Instant::now();
//     for _ in 0..1000 {
//         fft.forward(&a, &mut b)
//     }
//     let elapsed = now.elapsed();
//     dbg!(elapsed);
// }

fn main() {
    let mesh = Mesh::One(256);
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
    let mut rng = rand::thread_rng();
    system.sample_fields(&distr, &mut rng);

    // SCFT
    let config = SCFTConfig {
        steps: 100,
        step_size: 0.25,
        field_tolerance: 1e-5,
    };
    let simulation = SCFT::new(config);
    let result = simulation.run(&mut system);
    dbg!(result);
}
