use std::time::{Duration, Instant};

use crate::{simulation::FieldUpdater, system::System};

#[derive(Debug, Clone, Copy)]
pub struct SCFTParameters {
    pub dt: f64,
    pub etol: f64,
    pub nstep: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct SCFTState {
    pub f: f64,
    pub fbulk: f64,
    pub err: f64,
    pub step: usize,
    pub converged: bool,
    pub elapsed: Duration,
}

#[derive(Debug, Clone, Copy)]
pub struct SCFT {
    parameters: SCFTParameters,
}

impl SCFT {
    pub fn new(parameters: SCFTParameters) -> Self {
        Self { parameters }
    }

    pub fn run(&self, system: &mut System) -> SCFTState {
        let mut updater: FieldUpdater = FieldUpdater::new(system, self.parameters.dt);

        let start = Instant::now();
        for step in 0..self.parameters.nstep {
            let state = self.get_state(system, start, step);
            println!(
                "Step = {}, f = {:.5}, err = {:.5e}",
                state.step, state.f, state.err
            );
            if state.converged {
                return state;
            }
            updater.step(system);
        }

        self.get_state(system, start, self.parameters.nstep)
    }

    fn get_state(&self, system: &System, start: Instant, step: usize) -> SCFTState {
        let elapsed = start.elapsed();
        let err = system.field_error();
        let converged = err <= self.parameters.etol;
        SCFTState {
            f: system.free_energy(),
            fbulk: system.free_energy_bulk(),
            err,
            converged,
            step,
            elapsed,
        }
    }
}
