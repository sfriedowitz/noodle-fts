use ndarray::prelude::*;
use num::{complex::Complex64, Num};

type RField = Field<f64>;
type CField = Field<Complex64>;

struct Field<T: Copy + Num> {
    data: ArrayD<T>,
}

impl<T: Copy + Num> Field<T> {
    pub fn zeros(shape: &[usize]) -> Self {
        Self {
            data: ArrayD::<T>::zeros(IxDyn(shape)),
        }
    }
}
