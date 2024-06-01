use std::time::{Duration, Instant};

use crate::{simulation::FieldUpdater, system::System};

#[derive(Debug, Clone, Copy)]
pub struct SCFTConfig {
    pub steps: usize,
    pub step_size: f64,
    pub field_tolerance: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct SCFTState {
    pub step: usize,
    pub elapsed: Duration,
    pub is_converged: bool,
    pub field_error: f64,
    pub free_energy: f64,
    pub free_energy_bulk: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct SCFT {
    config: SCFTConfig,
}

impl SCFT {
    pub fn new(config: SCFTConfig) -> Self {
        Self { config }
    }

    pub fn run(&self, system: &mut System) -> SCFTState {
        let mut updater: FieldUpdater = FieldUpdater::new(system, self.config.step_size);

        let start = Instant::now();
        for step in 0..self.config.steps {
            let state = self.get_state(system, start, step);
            println!(
                "Step = {}, f = {:.5}, err = {:.5e}",
                state.step, state.free_energy, state.field_error
            );
            if state.is_converged {
                return state;
            }
            updater.step(system);
        }

        self.get_state(system, start, self.config.steps)
    }

    fn get_state(&self, system: &System, start: Instant, step: usize) -> SCFTState {
        let elapsed = start.elapsed();
        let field_error = system.field_error();
        let is_converged = field_error <= self.config.field_tolerance;
        SCFTState {
            step,
            elapsed,
            is_converged,
            field_error,
            free_energy: system.free_energy(),
            free_energy_bulk: system.free_energy_bulk(),
        }
    }
}
