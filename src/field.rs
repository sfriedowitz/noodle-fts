use ndarray::prelude::*;
use num::complex::Complex64;
use num::Num;

type RField = Field<f64>;
type CField = Field<Complex64>;

struct Field<T: Clone + Num> {
    data: ArrayD<T>,
}

impl<T: Clone + Num> Field<T> {
    pub fn zeros(shape: &[usize]) -> Self {
        Self {
            data: ArrayD::<T>::zeros(IxDyn(shape)),
        }
    }
}
