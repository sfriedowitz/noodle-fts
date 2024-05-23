use std::ops::Mul;

use itertools::Itertools;
use ndarray::Zip;
use ndarray_linalg::c64;
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

#[derive(Debug, Clone)]
struct PropagatorStep {
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
    fn new(mesh: Mesh) -> Self {
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

    fn apply(&mut self, q_in: &RField, q_out: &mut RField, method: StepMethod) {
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
        Self::apply_operator_inplace(&self.lw_full, q_in, &mut self.work_real);

        // Forward FFT into work_complex
        self.fft.forward(&self.work_real, &mut self.work_complex);

        // Apply lk_full to work_complex
        self.work_complex *= &self.lk_full;

        // Inverse FFT into work_real
        self.fft.inverse(&self.work_complex, &mut self.work_real);

        // Apply lw_full operator to work_real, store in q_full
        Self::apply_operator_inplace(&self.lw_full, &self.work_real, &mut self.q_full);
    }

    fn step_double_half(&mut self, q_in: &RField) {
        // Apply lw_half to q_in, store in work_real
        Self::apply_operator_inplace(&self.lw_half, q_in, &mut self.work_real);

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
        Self::apply_operator_inplace(&self.lw_half, &self.work_real, &mut self.q_double_half);
    }

    fn apply_operator_inplace<'a, T>(
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

#[derive(Debug, Clone)]
pub struct BlockPropagator {
    qs: Vec<RField>,
    step: PropagatorStep,
    step_size: f64,
    monomer_id: usize,
    monomer_size: f64,
    segment_length: f64,
}

impl BlockPropagator {
    pub fn new(
        mesh: Mesh,
        nstep: usize,
        step_size: f64,
        monomer_id: usize,
        monomer_size: f64,
        segment_length: f64,
    ) -> Self {
        let qs = (0..nstep + 1).map(|_| RField::zeros(mesh)).collect();
        let step = PropagatorStep::new(mesh);
        Self {
            qs,
            step,
            step_size,
            monomer_id,
            monomer_size,
            segment_length,
        }
    }

    pub fn qs(&self) -> &[RField] {
        &self.qs
    }

    pub fn update_operators(&mut self, fields: &[RField], ksq: &RField) {
        let field = &fields[self.monomer_id];
        let lw_coeff = self.step_size / 2.0;
        let ld_coeff = self.step_size * self.segment_length.powf(2.0) / 6.0;

        // Update w-operators
        Zip::from(&mut self.step.lw_full)
            .and(&mut self.step.lw_half)
            .and(field)
            .for_each(|full, half, omega| {
                *full = (-lw_coeff * self.monomer_size * omega).exp();
                *half = (-lw_coeff * self.monomer_size * omega / 2.0).exp();
            });

        // Update k-operators
        Zip::from(&mut self.step.lk_full)
            .and(&mut self.step.lk_half)
            .and(ksq)
            .for_each(|full, half, ksq| {
                let ksq_complex: Complex64 = ksq.into();
                *full = (-ld_coeff * ksq_complex).exp();
                *half = (-ld_coeff * ksq_complex / 2.0).exp();
            });
    }

    pub fn propagate(&mut self, q_initial: Option<&RField>) {
        if let Some(qi) = q_initial {
            self.qs[0].assign(qi);
        } else {
            self.qs[0].fill(1.0);
        }

        (1..self.qs.len()).for_each(|s| {
            let (left, right) = self.qs.split_at_mut(s);
            let q_in = &left[left.len() - 1];
            let q_out = &mut right[0];
            self.step.apply(q_in, q_out, StepMethod::RQM4);
        })
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn test() {
        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        let split = x.split_at_mut(1);
        dbg!(split);
    }
}
