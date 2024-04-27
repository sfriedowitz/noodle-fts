use ndarray::{ArrayD, IxDyn};
use num::complex::Complex64;
use rfts::fft::FFT;

pub fn main() {
    let mut fft = FFT::new(&[4, 4, 4, 4], None);

    let (nx, ny, nz, nw) = (4, 4, 4, 4);
    let mut input = ArrayD::<f64>::zeros(IxDyn(&[nx, ny, nz, nw]));
    for (i, v) in input.iter_mut().enumerate() {
        *v = i as f64;
    }

    let mut output = ArrayD::<Complex64>::zeros(IxDyn(&[nx, ny, nz, nw / 2 + 1]));

    fft.forward(&input, &mut output);
    fft.inverse(&output, &mut input);

    println!("{}", input);
}
