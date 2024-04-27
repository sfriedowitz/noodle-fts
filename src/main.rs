use ndarray::{ArrayD, IxDyn};
use num::complex::Complex64;

use rfts::fft::FFTPlan;

pub fn main() {
    let mut fft = FFTPlan::new(&[4, 4], None);

    let (nx, ny) = (4, 4);
    let mut data = ArrayD::<Complex64>::zeros(IxDyn(&[nx, ny]));
    let mut output = ArrayD::<Complex64>::zeros(IxDyn(&[nx, ny]));
    for (i, v) in data.iter_mut().enumerate() {
        *v = Complex64::new(i as f64, 0.0);
    }

    fft.forward(&data, &mut output);
    fft.inverse(&output, &mut data);

    println!("{}", data);
}
