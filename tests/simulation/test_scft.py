from datetime import timedelta

import pytest

from pynoodle import System
from pynoodle.simulation import SCFT, SCFTOptions, SCFTState


class TestSCFTOptions:
    def test_defaults(self):
        options = SCFTOptions()
        assert options.field_step_size == 0.1
        assert options.cell_step_size == 0.01
        assert options.variable_cell is False
        assert options.field_tolerance == 1e-4
        assert options.max_iterations == 1000

    def test_custom_options(self):
        options = SCFTOptions(
            field_step_size=0.05,
            variable_cell=True,
            max_iterations=500,
        )
        assert options.field_step_size == 0.05
        assert options.variable_cell is True
        assert options.max_iterations == 500


class TestSCFTState:
    def test_is_nan_detection(self):
        state = SCFTState(
            step=0,
            elapsed=timedelta(seconds=0),
            is_converged=False,
            field_error=float("nan"),
            stress_error=0.0,
            free_energy=0.0,
            free_energy_bulk=0.0,
        )
        assert state.is_nan is True

        state_valid = SCFTState(
            step=0,
            elapsed=timedelta(seconds=0),
            is_converged=False,
            field_error=1.0,
            stress_error=0.0,
            free_energy=0.0,
            free_energy_bulk=0.0,
        )
        assert state_valid.is_nan is False


class TestSCFT:
    @pytest.fixture
    def simple_system(self, mesh_1d, lamellar_cell, diblock_polymer):
        system = System(mesh=mesh_1d, cell=lamellar_cell, species=[diblock_polymer])
        system.set_interaction(0, 1, 0.5)
        system.sample_fields(scale=0.01, seed=42)
        return system

    def test_initialization(self, simple_system):
        solver = SCFT(simple_system)
        assert solver.system is simple_system
        assert solver.options.field_step_size == 0.1
        assert solver.field_updater is not None
        assert solver.cell_updater is None

    def test_initialization_with_variable_cell(self, simple_system):
        options = SCFTOptions(variable_cell=True)
        solver = SCFT(simple_system, options=options)
        assert solver.cell_updater is not None

    def test_run_few_iterations(self, simple_system):
        options = SCFTOptions(max_iterations=5)
        solver = SCFT(simple_system, options=options)
        state = solver.run(show_progress=False)

        assert isinstance(state, SCFTState)
        assert state.step <= 5
        assert not state.is_nan

    def test_convergence_detection(self, simple_system):
        options = SCFTOptions(field_tolerance=1e-2, max_iterations=100)
        solver = SCFT(simple_system, options=options)
        state = solver.run(show_progress=False)

        if state.is_converged:
            assert state.field_error <= options.field_tolerance
