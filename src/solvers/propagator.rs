use std::{cell::RefCell, ops::Mul, rc::Rc};

use ndarray::Zip;
use num::complex::Complex64;

use crate::{
    domain::{Mesh, FFT},
    fields::{CField, Field, RField},
};

#[derive(Debug, Clone, Copy)]
pub enum StepMethod {
    RK2,
    RQM4,
}

#[derive(Debug, Clone, Copy)]
pub enum PropagatorDirection {
    Forward,
    Reverse,
}

#[derive(Debug)]
pub struct PropagatorStep {
    fft: FFT,
    // Operator arrays
    lw_full: RField,
    lw_half: RField,
    lk_full: CField,
    lk_half: CField,
    // Ouptut arrays
    q_full: RField,
    q_double_half: RField,
    // Work arrays
    work_real: RField,
    work_complex: CField,
}

impl PropagatorStep {
    pub fn new(mesh: Mesh) -> Self {
        let kmesh = mesh.kmesh();
        Self {
            fft: FFT::new(mesh),
            lw_full: RField::zeros(mesh),
            lw_half: RField::zeros(mesh),
            lk_full: CField::zeros(kmesh),
            lk_half: CField::zeros(kmesh),
            q_full: RField::zeros(mesh),
            q_double_half: RField::zeros(mesh),
            work_real: RField::zeros(mesh),
            work_complex: CField::zeros(kmesh),
        }
    }

    pub fn update(
        &mut self,
        field: &RField,
        ksq: &RField,
        step_size: f64,
        monomer_size: f64,
        segment_length: f64,
    ) {
        let lw_coeff = step_size / 2.0;
        let lk_coeff = step_size * segment_length.powf(2.0) / 6.0;

        // Update w-operators:
        // Full = exp(-w * size * ds / 2)
        // Half = exp(-w * size * (ds / 2) / 2)
        Zip::from(&mut self.lw_full)
            .and(&mut self.lw_half)
            .and(field)
            .for_each(|full, half, w| {
                *full = (-lw_coeff * monomer_size * w).exp();
                *half = (-lw_coeff * monomer_size * w / 2.0).exp();
            });

        // Update k-operators:
        // Full = exp(-k^2 * b^2 * ds / 6)
        // Half = exp(-k^2 * b^2 * (ds / 2) / 6)
        Zip::from(&mut self.lk_full)
            .and(&mut self.lk_half)
            .and(ksq)
            .for_each(|full, half, ksq| {
                let ksq_complex: Complex64 = ksq.into();
                *full = (-lk_coeff * ksq_complex).exp();
                *half = (-lk_coeff * ksq_complex / 2.0).exp();
            });
    }

    pub fn apply(&mut self, q_in: &RField, q_out: &mut RField, method: StepMethod) {
        match method {
            StepMethod::RK2 => {
                // Full step to populate q_full
                self.step_full(q_in);
                // Copy into output array
                q_out.assign(&self.q_full)
            }
            StepMethod::RQM4 => {
                // Full step and double-half step to populate q_full and q_double_half
                self.step_full(q_in);
                self.step_double_half(q_in);
                // Richardson extrapolation of the results
                Zip::from(q_out)
                    .and(&self.q_full)
                    .and(&self.q_double_half)
                    .for_each(|output, qf, qdh| *output = (4.0 * qdh - qf) / 3.0);
            }
        }
    }

    fn step_full(&mut self, q_in: &RField) {
        // Apply lw_full to q_in, store in work_real
        Self::apply_operator_into(&self.lw_full, q_in, &mut self.work_real);

        // Forward FFT into work_complex
        self.fft.forward(&self.work_real, &mut self.work_complex);

        // Apply lk_full to work_complex
        self.work_complex *= &self.lk_full;

        // Inverse FFT into work_real
        self.fft.inverse(&self.work_complex, &mut self.work_real);

        // Apply lw_full operator to work_real, store in q_full
        Self::apply_operator_into(&self.lw_full, &self.work_real, &mut self.q_full);
    }

    fn step_double_half(&mut self, q_in: &RField) {
        // Apply lw_half to q_in, store in work_real
        Self::apply_operator_into(&self.lw_half, q_in, &mut self.work_real);

        // Forward FFT into work_complex
        self.fft.forward(&self.work_real, &mut self.work_complex);

        // Apply lk_half to work_complex
        self.work_complex *= &self.lk_half;

        // Inverse FFT into work_real
        self.fft.inverse(&self.work_complex, &mut self.work_real);

        // Apply lw_full to work_real
        self.work_real *= &self.lw_full;

        // Forward FFT into work_complex
        self.fft.forward(&self.work_real, &mut self.work_complex);

        // Apply lk_half to work_complex
        self.work_complex *= &self.lk_half;

        // Inverse FFT into work_real
        self.fft.inverse(&self.work_complex, &mut self.work_real);

        // Apply lw_half operator to work_real, store in q_double_half
        Self::apply_operator_into(&self.lw_half, &self.work_real, &mut self.q_double_half);
    }

    fn apply_operator_into<'a, T>(
        operator: &'a Field<T>,
        input: &'a Field<T>,
        output: &mut Field<T>,
    ) where
        &'a T: Mul<&'a T, Output = T>,
    {
        Zip::from(operator)
            .and(input)
            .and(output)
            .for_each(|operator, input, output| *output = operator * input)
    }
}

#[derive(Debug)]
pub struct Propagator {
    qfields: Vec<RField>,
    source: Option<Rc<RefCell<Propagator>>>,
}

impl Propagator {
    pub fn new(mesh: Mesh, ngrid: usize) -> Self {
        Self {
            qfields: (0..ngrid).map(|_| RField::zeros(mesh)).collect(),
            source: None,
        }
    }

    pub fn len(&self) -> usize {
        self.qfields.len()
    }

    pub fn qfields(&self) -> &[RField] {
        &self.qfields
    }

    pub fn first(&self) -> &RField {
        &self.qfields[0]
    }

    pub fn last(&self) -> &RField {
        &self.qfields[self.len() - 1]
    }

    pub fn add_source(&mut self, source: Rc<RefCell<Propagator>>) {
        self.source = Some(source)
    }

    pub fn propagate(&mut self, step: &mut PropagatorStep) {
        // Initial condition depending on whether a source is present
        if let Some(source) = &self.source {
            self.qfields[0].assign(source.borrow().last())
        } else {
            self.qfields[0].fill(1.0)
        }
        // Propagate from 1..len()
        for s in 1..self.len() {
            let (head, tail) = self.qfields.split_at_mut(s);
            let q_in = head.last().unwrap();
            let q_out = tail.first_mut().unwrap();
            step.apply(q_in, q_out, StepMethod::RQM4)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use ndarray_rand::{rand_distr::Normal, RandomExt};
    use propagator::Propagator;

    use crate::{
        domain::{Domain, Mesh, UnitCell},
        fields::RField,
        solvers::{propagator, PropagatorStep},
    };

    #[test]
    fn test_propagator() {
        let mesh = Mesh::One(128);
        let cell = UnitCell::lamellar(10.0).unwrap();
        let domain = Domain::new(mesh, cell).unwrap();

        let field = RField::random(mesh, Normal::new(0.0, 0.1).unwrap());
        let ksq = domain.ksq().unwrap();

        let mut propagator = Propagator::new(mesh, 100);
        let mut step = PropagatorStep::new(mesh);
        step.update(&field, &ksq, 0.01, 1.0, 1.0);

        let now = Instant::now();
        propagator.propagate(&mut step);
        let elapsed = now.elapsed();

        // dbg!(propagator);
        dbg!(elapsed);
    }
}
