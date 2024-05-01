use ndarray::{array, Array2};
use ndarray_linalg::Inverse;

#[derive(Debug, Clone, Copy)]
pub enum CellParameters {
    OneD(f64),
    TwoD {
        a: f64,
        b: f64,
        theta: f64,
    },
    ThreeD {
        a: f64,
        b: f64,
        c: f64,
        alpha: f64,
        beta: f64,
        gamma: f64,
    },
}

pub struct UnitCell {
    parameters: CellParameters,
    cell: Array2<f64>,
    cell_inv: Array2<f64>,
    metric: Array2<f64>,
    metric_inv: Array2<f64>,
}

impl UnitCell {
    pub fn new(parameters: CellParameters) -> Self {
        let cell = match parameters {
            CellParameters::OneD(length) => UnitCell::make_1d(length),
            CellParameters::TwoD { a, b, theta } => UnitCell::make_2d(a, b, theta),
            CellParameters::ThreeD {
                a,
                b,
                c,
                alpha,
                beta,
                gamma,
            } => UnitCell::make_3d(a, b, c, alpha, beta, gamma),
        };
        let cell_inv = cell.inv().unwrap();
        let metric = cell.dot(&cell.t());
        let metric_inv = metric.inv().unwrap();

        Self {
            parameters,
            cell,
            cell_inv,
            metric,
            metric_inv,
        }
    }

    pub fn scale_real(&self) {
        todo!()
    }

    pub fn scale_reciprocal(&self) {
        todo!()
    }

    fn make_1d(length: f64) -> Array2<f64> {
        array![[length]]
    }

    fn make_2d(a: f64, b: f64, theta: f64) -> Array2<f64> {
        array![[a]]
    }

    fn make_3d(a: f64, b: f64, c: f64, alpha: f64, beta: f64, theta: f64) -> Array2<f64> {
        array![[a]]
    }
}
