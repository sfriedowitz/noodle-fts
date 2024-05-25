use std::ops::Mul;

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

    pub fn update_operators(
        &mut self,
        field: &RField,
        ksq: &RField,
        segment_length: f64,
        monomer_size: f64,
        ds: f64,
    ) {
        let lw_coeff = ds / 2.0;
        let lk_coeff = ds * segment_length.powf(2.0) / 6.0;

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
        Self::multiply_operator_into(&self.lw_full, q_in, &mut self.work_real);

        // Forward FFT into work_complex
        self.fft.forward(&self.work_real, &mut self.work_complex);

        // Apply lk_full to work_complex
        self.work_complex *= &self.lk_full;

        // Inverse FFT into work_real
        self.fft.inverse(&self.work_complex, &mut self.work_real);

        // Apply lw_full operator to work_real, store in q_full
        Self::multiply_operator_into(&self.lw_full, &self.work_real, &mut self.q_full);
    }

    fn step_double_half(&mut self, q_in: &RField) {
        // Apply lw_half to q_in, store in work_real
        Self::multiply_operator_into(&self.lw_half, q_in, &mut self.work_real);

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
        Self::multiply_operator_into(&self.lw_half, &self.work_real, &mut self.q_double_half);
    }

    fn multiply_operator_into<'a, T>(
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
}

impl Propagator {
    pub fn new(mesh: Mesh, ns: usize) -> Self {
        Self {
            qfields: (0..ns).map(|_| RField::zeros(mesh)).collect(),
        }
    }

    pub fn ns(&self) -> usize {
        self.qfields.len()
    }

    pub fn head(&self) -> &RField {
        &self.qfields[0]
    }

    pub fn head_mut(&mut self) -> &mut RField {
        &mut self.qfields[0]
    }

    pub fn tail(&self) -> &RField {
        &self.qfields[self.ns() - 1]
    }

    pub fn tail_mut(&mut self) -> &mut RField {
        let s = self.ns() - 1;
        &mut self.qfields[s]
    }

    pub fn position(&self, s: usize) -> &RField {
        &self.qfields[s]
    }

    pub fn position_mut(&mut self, s: usize) -> &mut RField {
        &mut self.qfields[s]
    }

    pub fn propagate(&mut self, step: &mut PropagatorStep) {
        // Propagate from 1..ns
        for s in 1..self.ns() {
            let (left, right) = self.qfields.split_at_mut(s);
            let q_in = left.last().unwrap();
            let q_out = right.first_mut().unwrap();
            step.apply(q_in, q_out, StepMethod::RQM4)
        }
    }

    pub fn update_head<'a>(&mut self, sources: impl Iterator<Item = &'a RField>) {
        self.head_mut().fill(1.0);
        for source in sources {
            *self.head_mut() *= source;
        }
    }
}
