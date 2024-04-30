mod chem;
mod domain;
mod interactions;
mod species;
mod system;

#[cfg(test)]
mod tests {
    use crate::domain::{fft::FFT, mesh::Mesh, CField, RField};

    #[test]
    fn testing() {
        let mesh = Mesh::new(vec![4, 4]);

        let mut fft = FFT::new(&mesh, None);

        let mut input = RField::zeros(mesh.dimensions());
        for (i, v) in input.iter_mut().enumerate() {
            *v = i as f64;
        }

        let mut output = CField::zeros(mesh.to_complex().dimensions());

        fft.forward(&input, &mut output);
        fft.inverse(&output, &mut input);

        println!("{}", input);
        println!("{}", output);
    }
}
