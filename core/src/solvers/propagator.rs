use crate::{
    domain::{Mesh, FFT},
    fields::{CField, FieldExt, RField},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepMethod {
    RK2,
    RQM4,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PropagatorDirection {
    Forward,
    Reverse,
}

#[derive(Debug)]
pub struct PropagatorStep {
    fft: FFT,
    // Operator arrays
    lw1: RField,
    lw2: RField,
    lk1: CField,
    lk2: CField,
    // Ouptut arrays
    q1: RField,
    q2: RField,
    // Work arrays
    qr: RField,
    qk: CField,
}

impl PropagatorStep {
    pub fn new(mesh: Mesh) -> Self {
        let kmesh = mesh.kmesh();
        Self {
            fft: FFT::new(mesh),
            lw1: RField::zeros(mesh),
            lw2: RField::zeros(mesh),
            lk1: CField::zeros(kmesh),
            lk2: CField::zeros(kmesh),
            q1: RField::zeros(mesh),
            q2: RField::zeros(mesh),
            qr: RField::zeros(mesh),
            qk: CField::zeros(kmesh),
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
        self.lw1
            .zip_mut_with(field, |op, w| *op = (-lw_coeff * monomer_size * w).exp());
        self.lw2
            .zip_mut_with(field, |op, w| *op = (-lw_coeff * monomer_size * w / 2.0).exp());

        // Update k-operators:
        // Full = exp(-k^2 * b^2 * ds / 6)
        // Half = exp(-k^2 * b^2 * (ds / 2) / 6)
        self.lk1
            .zip_mut_with(ksq, |op, k2| *op = (-lk_coeff * k2).exp().into());
        self.lk2
            .zip_mut_with(ksq, |op, k2| *op = (-lk_coeff * k2 / 2.0).exp().into());
    }

    pub fn apply(&mut self, q_in: &RField, q_out: &mut RField, method: StepMethod) {
        match method {
            StepMethod::RK2 => {
                // Full step to populate q1
                self.step_full(q_in);
                // Copy into output array
                q_out.assign(&self.q1)
            }
            StepMethod::RQM4 => {
                // Full step and double-half step to populate q1 and q2
                self.step_full(q_in);
                self.step_double_half(q_in);
                // Richardson extrapolation of the results
                q_out.zip_mut_with_two(&self.q1, &self.q2, |out, full, half| {
                    *out = (4.0 * half - full) / 3.0
                });
            }
        }
    }

    fn step_full(&mut self, q_in: &RField) {
        // Apply lw1 to q_in, store in qr
        self.qr.zip_mut_with_two(&self.lw1, q_in, |z, x, y| *z = x * y);

        // Forward FFT into qk
        self.fft.forward(&self.qr, &mut self.qk);

        // Apply lk1 to qk
        self.qk *= &self.lk1;

        // Inverse FFT into qr
        self.fft.inverse(&self.qk, &mut self.qr);

        // Apply lw1 operator to qr, store in q1
        self.q1
            .zip_mut_with_two(&self.lw1, &self.qr, |z, x, y| *z = x * y);
    }

    fn step_double_half(&mut self, q_in: &RField) {
        // Apply lw2 to q_in, store in qr
        self.qr.zip_mut_with_two(&self.lw2, q_in, |z, x, y| *z = x * y);

        // Forward FFT into qk
        self.fft.forward(&self.qr, &mut self.qk);

        // Apply lk2 to qk
        self.qk *= &self.lk2;

        // Inverse FFT into qr
        self.fft.inverse(&self.qk, &mut self.qr);

        // Apply lw1 to qr
        self.qr *= &self.lw1;

        // Forward FFT into qk
        self.fft.forward(&self.qr, &mut self.qk);

        // Apply lk2 to qk
        self.qk *= &self.lk2;

        // Inverse FFT into qr
        self.fft.inverse(&self.qk, &mut self.qr);

        // Apply lk2 operator to qr, store in q2
        self.q2
            .zip_mut_with_two(&self.lw2, &self.qr, |z, x, y| *z = x * y);
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

    pub fn update_head<'a>(&mut self, sources: impl Iterator<Item = &'a RField>) {
        self.head_mut().fill(1.0);
        for source in sources {
            *self.head_mut() *= source;
        }
    }

    pub fn propagate(&mut self, step: &mut PropagatorStep, method: StepMethod) {
        for s in 1..self.ns() {
            let (left, right) = self.qfields.split_at_mut(s);
            let q_in = &left[left.len() - 1];
            let q_out = &mut right[0];
            step.apply(q_in, q_out, method)
        }
    }
}
