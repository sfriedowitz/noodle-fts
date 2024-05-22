use ndarray::Zip;

use crate::{
    domain::{Mesh, FFT},
    fields::{CField, RField},
    math::{apply_operator, apply_operator_inplace},
};

#[derive(Debug, Clone, Copy)]
pub enum StepMethod {
    RK2,
    RQM4,
}

#[derive(Debug)]
struct PropagatorStep {
    // Method
    method: StepMethod,
    // Operator arrays
    lw1: RField,
    lw2: RField,
    lk1: RField,
    lk2: RField,
    // Work arrays
    q_full: RField,
    q_double_half: RField,
    work_real: RField,
    work_complex: CField,
}

impl PropagatorStep {
    fn new(mesh: Mesh, method: StepMethod) -> Self {
        let kmesh = mesh.kmesh();
        Self {
            method,
            lw1: RField::zeros(mesh),
            lw2: RField::zeros(mesh),
            lk1: RField::zeros(kmesh),
            lk2: RField::zeros(kmesh),
            q_full: RField::zeros(mesh),
            q_double_half: RField::zeros(mesh),
            work_real: RField::zeros(mesh),
            work_complex: CField::zeros(kmesh),
        }
    }

    fn update(
        &mut self,
        field: &RField,
        ksq: &RField,
        monomer_size: f64,
        segment_length: f64,
        ds: f64,
    ) {
        todo!()
    }

    fn step(&mut self, fft: &mut FFT, qin: &RField, qout: &mut RField) {
        match self.method {
            StepMethod::RK2 => {
                // Full step into q_full
                self.step_full(fft, qin);
                // Copy into output array
                qout.assign(&self.q_full)
            }
            StepMethod::RQM4 => {
                // Full step and double-half step to populate q_full and q_double_half
                self.step_full(fft, qin);
                self.step_double_half(fft, qin);
                // Richardson extrapolation of the results
                Zip::from(qout)
                    .and(&self.q_full)
                    .and(&self.q_double_half)
                    .for_each(|output, qf, qdh| *output = (4.0 * qdh - qf) / 3.0);
            }
        }
    }

    fn step_full(&mut self, fft: &mut FFT, qin: &RField) {
        // Apply lw1 operator to qin, store in work_real
        apply_operator(&self.lw1, qin, &mut self.work_real);

        // Forward FFT into work_complex
        fft.forward(&self.work_real, &mut self.work_complex);

        // Apply lk1 to work_complex
        apply_operator_inplace(&self.lk1, &mut self.work_complex);

        // Inverse FFT into work_real
        fft.inverse(&self.work_complex, &mut self.work_real);

        // Apply lw1 operator to work_real, store in q_full
        apply_operator(&self.lw1, &self.work_real, &mut self.q_full);
    }

    fn step_double_half(&mut self, fft: &mut FFT, qin: &RField) {
        // Apply lw2 operator to qin, store in work_real
        apply_operator(&self.lw2, qin, &mut self.work_real);

        // Forward FFT into work_complex
        fft.forward(&self.work_real, &mut self.work_complex);

        // Apply lk2 to work_complex
        apply_operator_inplace(&self.lk2, &mut self.work_complex);

        // Inverse FFT into work_real
        fft.inverse(&self.work_complex, &mut self.work_real);

        // Apply lw1 to work_real
        apply_operator_inplace(&self.lw1, &mut self.work_real);

        // Forward FFT into work_complex
        fft.forward(&self.work_real, &mut self.work_complex);

        // Apply lk2 to work_complex
        apply_operator_inplace(&self.lk2, &mut self.work_complex);

        // Inverse FFT into work_real
        fft.inverse(&self.work_complex, &mut self.work_real);

        // Apply lw2 operator to work_real, store in q_double_half
        apply_operator(&self.lw2, &self.work_real, &mut self.q_double_half);
    }
}

#[derive(Debug)]
pub struct BlockPropagator {
    fft: FFT,
    step: PropagatorStep,
    solutions: Vec<RField>,
    step_size: f64,
    monomer_size: f64,
    segment_length: f64,
}

impl BlockPropagator {
    pub fn new(
        mesh: Mesh,
        nstep: usize,
        step_size: f64,
        monomer_size: f64,
        segment_length: f64,
    ) -> Self {
        todo!()
    }
}
