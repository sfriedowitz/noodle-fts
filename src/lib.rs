mod dimensions;
mod fft;
mod mde;
mod system;
mod types;

#[cfg(test)]
mod tests {
    use ndarray::IxDyn;

    use crate::{
        fft::FFT,
        types::{CField, RField},
    };

    #[test]
    fn testing() {
        let mut fft = FFT::new(&[4], None);

        let nx = 4;
        let mut input = RField::zeros(IxDyn(&[nx]));
        for (i, v) in input.iter_mut().enumerate() {
            *v = i as f64;
        }

        let mut output = CField::zeros(IxDyn(&[nx / 2 + 1]));

        fft.forward(&input, &mut output);
        fft.inverse(&output, &mut input);

        println!("{}", input);
        println!("{}", output);
    }
}
