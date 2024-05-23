use std::ops::Mul;

use ndarray::Zip;

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
pub(super) struct PropagatorStep {
    fft: FFT,
    // Operator arrays
    pub lw_full: RField,
    pub lw_half: RField,
    pub lk_full: CField,
    pub lk_half: CField,
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
