use ndarray::{ArrayD, Zip};
use num::complex::Complex64;

/// A multi-dimensional array with dynamic shape.
pub type Field<T> = ArrayD<T>;

/// Real-valued multi-dimensional field grid.
pub type RField = Field<f64>;

/// Complex-valued multi-dimensional field grid.
pub type CField = Field<Complex64>;

pub trait FieldOps<A> {
    /// Accumulate `f(acc, self, other)` from the initial value `init`.
    fn fold_with<B, C, F>(&self, other: &Field<B>, init: C, f: F) -> C
    where
        F: Fn(C, &A, &B) -> C;

    /// Traverse three arrays in unspecified order, in lock step,
    /// calling the closure `f` on each triplet of elements.
    fn zip_mut_with_two<B, C, F>(&mut self, first: &Field<B>, second: &Field<C>, f: F)
    where
        F: FnMut(&mut A, &B, &C);
}

impl<A> FieldOps<A> for Field<A> {
    fn fold_with<B, C, F>(&self, other: &Field<B>, init: C, f: F) -> C
    where
        F: Fn(C, &A, &B) -> C,
    {
        Zip::from(self).and(other).fold(init, f)
    }

    fn zip_mut_with_two<B, C, F>(&mut self, first: &Field<B>, second: &Field<C>, f: F)
    where
        F: FnMut(&mut A, &B, &C),
    {
        Zip::from(self).and(first).and(second).for_each(f)
    }
}
