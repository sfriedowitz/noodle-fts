use std::fmt;

use ndrustfft::{ndfft, ndfft_r2c, ndifft, ndifft_r2c, FftHandler, R2cFftHandler};

use super::Mesh;
use crate::fields::{CField, RField};

/// Wrapper for real-to-complex FFTs over a multi-dimensional array.
/// The real-to-complex transformation is performed over the last axis of the arrays.
#[derive(Clone)]
pub struct FFT {
    // 1 real, ndim-1 complex handlers
    r2c_handler: R2cFftHandler<f64>,
    c2c_handlers: Vec<FftHandler<f64>>,
    // Work buffers with half-size last dimension
    work1: CField,
    work2: CField,
}

impl FFT {
    pub fn new(mesh: Mesh) -> Self {
        // This creates a half-dimension r2c handler when providing the full value of n
        let r2c_handler = match mesh {
            Mesh::One(nx) => R2cFftHandler::<f64>::new(nx),
            Mesh::Two(_, ny) => R2cFftHandler::<f64>::new(ny),
            Mesh::Three(_, _, nz) => R2cFftHandler::<f64>::new(nz),
        };

        let c2c_handlers = mesh
            .dimensions()
            .iter()
            .take(mesh.ndim() - 1)
            .map(|dim| FftHandler::<f64>::new(*dim))
            .collect();

        let kmesh = mesh.kmesh();
        let work1 = CField::zeros(kmesh);
        let work2 = CField::zeros(kmesh);

        Self {
            r2c_handler,
            c2c_handlers,
            work1,
            work2,
        }
    }

    pub fn ndim(&self) -> usize {
        self.c2c_handlers.len() + 1
    }

    pub fn forward(&mut self, input: &RField, output: &mut CField) {
        // Transform the real -> complex dimension first
        let r2c_axis = self.ndim() - 1;
        ndfft_r2c(input, output, &mut self.r2c_handler, r2c_axis);

        // Transform the remaining complex -> complex axes
        self.work1.assign(output);
        for (axis, handler) in self.c2c_handlers.iter_mut().enumerate() {
            ndfft(&self.work1, output, handler, axis);
            self.work1.assign(output);
        }
    }

    pub fn inverse(&mut self, input: &CField, output: &mut RField) {
        // Transform the complex -> complex axes first
        self.work1.assign(input);
        self.work2.assign(input);
        for (axis, handler) in self.c2c_handlers.iter_mut().enumerate() {
            ndifft(&self.work1, &mut self.work2, handler, axis);
            self.work1.assign(&self.work2);
        }

        // Transform the accumulated work into the output array
        let r2c_axis = self.ndim() - 1;
        ndifft_r2c(&self.work2, output, &mut self.r2c_handler, r2c_axis);
    }
}

impl fmt::Debug for FFT {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FFT").field("ndim", &self.ndim()).finish()
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        domain::{Mesh, FFT},
        fields::{CField, RField},
    };

    #[test]
    fn test_fft() {
        let meshes = vec![Mesh::One(4), Mesh::Two(4, 4), Mesh::Three(4, 4, 4)];

        meshes.iter().cloned().for_each(|mesh| {
            let mut fft = FFT::new(mesh);

            let mut input = RField::zeros(mesh);
            let mut output = RField::zeros(mesh);
            for (i, v) in input.iter_mut().enumerate() {
                *v = i as f64;
            }
            let mut work = CField::zeros(mesh.kmesh());

            // Run a forward and inverse FFT on the input
            fft.forward(&input, &mut work);
            fft.inverse(&work, &mut output);

            // Input and output should be approx equal after round-trip FFT
            assert!(input.abs_diff_eq(&output, 1e-8));
        });
    }
}
