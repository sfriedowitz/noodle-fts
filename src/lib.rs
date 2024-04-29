mod fft;
mod mde;
mod mesh;
mod species;
mod system;
mod types;

#[cfg(test)]
mod tests {
    use ndarray::IxDyn;

    use crate::{
        fft::FFT,
        mesh::Mesh,
        types::{CField, RField},
    };

    #[test]
    fn testing() {
        let mesh = Mesh::new(&[4, 4]);

        let mut fft = FFT::new(mesh.clone(), None);

        let mut input = RField::zeros(IxDyn(mesh.dimensions()));
        for (i, v) in input.iter_mut().enumerate() {
            *v = i as f64;
        }

        let mut output = CField::zeros(IxDyn(mesh.k_dimensions()));

        fft.forward(&input, &mut output);
        fft.inverse(&output, &mut input);

        println!("{}", input);
        println!("{}", output);
    }
}
