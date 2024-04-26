use ndarray::{ArrayD, IxDyn};
use ndrustfft::{ndfft, Complex, FftHandler};

pub fn main() {
    let (nx, ny) = (4, 4);
    let mut data = ArrayD::<Complex<f64>>::zeros(IxDyn(&[nx, ny]));
    let mut output = ArrayD::<Complex<f64>>::zeros(IxDyn(&[nx, ny]));
    for (i, v) in data.iter_mut().enumerate() {
        *v = Complex::new(i as f64, 0.0);
    }

    let mut fft_x = FftHandler::<f64>::new(nx);
    let mut fft_y = FftHandler::<f64>::new(ny);

    let mut work = ArrayD::<Complex<f64>>::zeros(IxDyn(&[nx, ny]));
    ndfft(&data.view(), &mut work.view_mut(), &mut fft_x, 0);
    ndfft(&work.view(), &mut output.view_mut(), &mut fft_y, 1);

    println!("{}", output);
}
