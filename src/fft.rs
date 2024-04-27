use ndarray::{ArrayD, IxDyn};
use ndrustfft::{ndfft, ndfft_r2c, ndifft, ndifft_r2c, FftHandler, Normalization, R2cFftHandler};
use num::complex::Complex64;

/// Wrapper for real-to-complex FFTs over a multi-dimensional array.
/// The real-to-complex transformation is performed over the last axis of the arrays.
pub struct FFT {
    dimensions: Vec<usize>,
    r2c_handler: R2cFftHandler<f64>,
    c2c_handlers: Vec<FftHandler<f64>>,
    // Work buffers have half-size last dimension
    work1: ArrayD<Complex64>,
    work2: ArrayD<Complex64>,
}

impl FFT {
    pub fn new(dimensions: &[usize], normalization: Option<Normalization<Complex64>>) -> Self {
        let r2c_dim = dimensions.last().unwrap();
        let r2c_handler = match &normalization {
            Some(norm) => R2cFftHandler::<f64>::new(*r2c_dim).normalization(norm.clone()),
            None => R2cFftHandler::<f64>::new(*r2c_dim),
        };

        let c2c_handlers = dimensions
            .iter()
            .take(dimensions.len() - 1)
            .map(|d| match &normalization {
                Some(norm) => FftHandler::<f64>::new(*d).normalization(norm.clone()),
                None => FftHandler::<f64>::new(*d),
            })
            .collect();

        let kdimensions = Self::complex_dimensions(dimensions);
        let work1 = ArrayD::<Complex64>::zeros(IxDyn(&kdimensions));
        let work2 = ArrayD::<Complex64>::zeros(IxDyn(&kdimensions));

        Self {
            dimensions: dimensions.to_vec(),
            r2c_handler,
            c2c_handlers,
            work1,
            work2,
        }
    }

    pub fn complex_dimensions(dimensions: &[usize]) -> Vec<usize> {
        let mut kdims = dimensions.to_vec();
        let r2c_dim = kdims.last_mut().unwrap();
        *r2c_dim = *r2c_dim / 2 + 1;
        kdims
    }

    pub fn n_dimensions(&self) -> usize {
        self.dimensions.len()
    }

    pub fn forward(&mut self, input: &ArrayD<f64>, output: &mut ArrayD<Complex64>) {
        // Transform the real -> complex dimension first
        let r2c_axis = self.n_dimensions() - 1;
        ndfft_r2c(input, output, &mut self.r2c_handler, r2c_axis);

        // Transform the remaining complex -> complex axes
        self.work1.assign(output);
        for (c2c_axis, c2c_handler) in self.c2c_handlers.iter_mut().enumerate() {
            ndfft(&self.work1, output, c2c_handler, c2c_axis);
            self.work1.assign(output);
        }
    }

    pub fn inverse(&mut self, input: &ArrayD<Complex64>, output: &mut ArrayD<f64>) {
        // Transform the complex -> complex axes first
        self.work1.assign(input);
        for (c2c_axis, c2c_handler) in self.c2c_handlers.iter_mut().enumerate() {
            ndifft(&self.work1, &mut self.work2, c2c_handler, c2c_axis);
            self.work1.assign(&self.work2);
        }

        // Transform the accumulated work into the output array
        let r2c_axis = self.n_dimensions() - 1;
        ndifft_r2c(&self.work2, output, &mut self.r2c_handler, r2c_axis);
    }
}
