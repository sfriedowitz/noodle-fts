use std::{borrow::Borrow, collections::HashMap};

use ndarray::{ArrayD, IxDyn};
use ndrustfft::{ndfft, ndifft, FftHandler, Normalization};
use num::complex::Complex64;

pub struct FFTPlan {
    handlers: HashMap<usize, FftHandler<f64>>,
    work: ArrayD<Complex64>,
}

impl FFTPlan {
    pub fn new(dimensions: &[usize], normalization: Option<Normalization<Complex64>>) -> Self {
        let handlers = dimensions
            .iter()
            .enumerate()
            .map(|(i, d)| {
                let h = match normalization.borrow() {
                    Some(norm) => FftHandler::<f64>::new(*d).normalization(norm.clone()),
                    None => FftHandler::<f64>::new(*d),
                };
                (i, h)
            })
            .collect();

        let work = ArrayD::<Complex64>::zeros(IxDyn(dimensions));

        Self { handlers, work }
    }

    pub fn forward(&mut self, x: &ArrayD<Complex64>, xhat: &mut ArrayD<Complex64>) {
        self.work.assign(x);
        for (axis, handler) in self.handlers.iter_mut() {
            ndfft(&self.work, xhat, handler, *axis);
            self.work.assign(xhat);
        }
    }

    pub fn inverse(&mut self, xhat: &ArrayD<Complex64>, x: &mut ArrayD<Complex64>) {
        self.work.assign(xhat);
        for (axis, handler) in self.handlers.iter_mut() {
            ndifft(&self.work, x, handler, *axis);
            self.work.assign(x);
        }
    }
}
