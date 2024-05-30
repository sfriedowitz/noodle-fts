use std::ops::AddAssign;

use ndarray::{ArrayD, Zip};
use num::complex::Complex64;

/// A multi-dimensional array with dynamic shape.
pub type Field<T> = ArrayD<T>;

/// Real-valued multi-dimensional field grid.
pub type RField = Field<f64>;

/// Complex-valued multi-dimensional field grid.
pub type CField = Field<Complex64>;

pub trait FieldOps<A>
where
    A: AddAssign<A>,
{
    /// Perform the fold with accumulation defined by `acc + f(self, other)`.
    fn fold_with<B, C, F: FnMut(C, &A, &B) -> C>(&self, init: C, other: &Field<B>, f: F) -> C;

    /// Perform the operation `self = f(other)` elementwise.
    fn assign_unary_fn<B, F: Fn(&B) -> A>(&mut self, other: &Field<B>, f: F);

    /// Perform the operation `self = f(first, second)` elementwise.
    fn assign_binary_fn<B, C, F: Fn(&B, &C) -> A>(
        &mut self,
        first: &Field<B>,
        second: &Field<C>,
        f: F,
    );

    /// Perform the operation `self += f(other)` elementwise.
    fn add_assign_unary_fn<B, F: Fn(&B) -> A>(&mut self, other: &Field<B>, f: F);

    /// Perform the operation `self += f(first, second)` elementwise.
    fn add_assign_binary_fn<B, C, F: Fn(&B, &C) -> A>(
        &mut self,
        first: &Field<B>,
        second: &Field<C>,
        f: F,
    );
}

impl<A> FieldOps<A> for Field<A>
where
    A: AddAssign<A>,
{
    fn fold_with<B, C, F: FnMut(C, &A, &B) -> C>(&self, init: C, other: &Field<B>, f: F) -> C {
        Zip::from(self).and(other).fold(init, f)
    }

    fn assign_unary_fn<B, F: Fn(&B) -> A>(&mut self, other: &Field<B>, f: F) {
        Zip::from(self).and(other).for_each(|y, x| *y = f(x));
    }

    fn assign_binary_fn<B, C, F: Fn(&B, &C) -> A>(
        &mut self,
        first: &Field<B>,
        second: &Field<C>,
        f: F,
    ) {
        Zip::from(self)
            .and(first)
            .and(second)
            .for_each(|z, x, y| *z = f(x, y));
    }

    fn add_assign_unary_fn<B, F: Fn(&B) -> A>(&mut self, other: &Field<B>, f: F) {
        Zip::from(self).and(other).for_each(|y, x| *y += f(x));
    }

    fn add_assign_binary_fn<B, C, F: Fn(&B, &C) -> A>(
        &mut self,
        first: &Field<B>,
        second: &Field<C>,
        f: F,
    ) {
        Zip::from(self)
            .and(first)
            .and(second)
            .for_each(|z, x, y| *z += f(x, y));
    }
}
