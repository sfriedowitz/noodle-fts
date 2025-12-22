from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
from tqdm import tqdm

from pynoodle import CellUpdater, FieldUpdater, System


@dataclass
class SCFTOptions:
    """Configuration options for SCFT solver.

    Attributes:
        field_step_size: Step size for field updates (default: 0.1)
        cell_step_size: Step size for cell parameter updates (default: 0.01)
        variable_cell: Whether to enable variable cell optimization (default: False)
        field_tolerance: Convergence tolerance for field error (default: 1e-6)
        stress_tolerance: Convergence tolerance for stress error (default: 1e-3)
        max_iterations: Maximum number of SCFT iterations (default: 1000)
    """

    field_step_size: float = 0.1
    cell_step_size: float = 0.01
    variable_cell: bool = False
    field_tolerance: float = 1e-4
    stress_tolerance: float = 1e-3
    max_iterations: int = 1000


@dataclass
class SCFTState:
    """State information from SCFT iteration.

    Attributes:
        step: Current iteration number
        elapsed: Time elapsed since start
        is_converged: Whether convergence criteria has been met
        field_error: Current field error
        stress_error: Current stress tensor error (Frobenius norm)
        free_energy: Current free energy
        free_energy_bulk: Current bulk free energy
    """

    step: int
    elapsed: timedelta
    is_converged: bool
    field_error: float
    stress_error: float
    free_energy: float
    free_energy_bulk: float

    @property
    def is_nan(self) -> bool:
        """Check if any state values are NaN."""
        return np.any(
            np.isnan(
                [self.field_error, self.stress_error, self.free_energy, self.free_energy_bulk]
            )
        )


class SCFT:
    """Self-consistent field theory (SCFT) solver.

    This class manages the SCFT iteration process, including field updates
    and optional variable cell optimization. It creates and manages the
    necessary updaters internally based on the provided options.

    Cell updates are performed at a fixed interval relative to field updates.
    For example, with cell_update_interval=10, the cell is updated every 10
    field update steps. This prevents expensive cell updates from running too
    frequently while fields are still equilibrating.

    Args:
        system: The polymer system to solve
        options: Configuration options (uses defaults if not provided)

    Example:
        >>> system = System(mesh, cell, species)
        >>> options = SCFTOptions(
        ...     field_step_size=0.1,
        ...     cell_step_size=0.1,
        ...     variable_cell=True,
        ...     max_iterations=500
        ... )
        >>> solver = SCFT(system, options)
        >>> result = solver.run()
    """

    def __init__(
        self,
        system: System,
        *,
        options: SCFTOptions | None = None,
    ):
        """Initialize SCFT solver.

        Args:
            system: The polymer system to solve
            options: Configuration options (uses defaults if not provided)
        """
        self.system = system
        self.options = options or SCFTOptions()

        # Create field updater
        self.field_updater = FieldUpdater(system, self.options.field_step_size)

        # Create cell updater if enabled
        self.cell_updater = (
            CellUpdater(self.options.cell_step_size) if self.options.variable_cell else None
        )

    def run(self, *, show_progress: bool = True) -> SCFTState:
        """Run SCFT iterations until convergence or maximum iterations.

        Args:
            show_progress: Whether to display progress bar (default: True)

        Returns:
            SCFTState containing final iteration information

        Raises:
            RuntimeError: If system state contains NaN values during iteration
        """
        start = datetime.now()
        iterator = range(self.options.max_iterations)
        pbar = tqdm(iterator, desc="SCFT") if show_progress else iterator

        for step in pbar:
            state = self._get_state(start, step)

            if show_progress:
                postfix = {"f": state.free_energy, "field_err": state.field_error}
                if self.options.variable_cell:
                    postfix["stress_err"] = state.stress_error
                pbar.set_postfix(postfix)

            if state.is_nan:
                raise RuntimeError("System state contains NaN values -- breaking.")

            if state.is_converged:
                return state

            # Perform field update
            self.field_updater.step(self.system)

            # Perform cell update at specified interval
            if self.cell_updater is not None:
                self.cell_updater.step(self.system)

        return self._get_state(start, self.options.max_iterations)

    def _get_state(self, start: datetime, step: int) -> SCFTState:
        """Get current state of the SCFT iteration."""
        elapsed = datetime.now() - start
        field_error = self.system.field_error()
        stress_error = self.system.stress_error()

        # Check convergence: fields must always converge
        # If variable_cell is enabled, stress must also converge
        field_converged = field_error <= self.options.field_tolerance
        stress_converged = (
            stress_error <= self.options.stress_tolerance if self.options.variable_cell else True
        )
        is_converged = field_converged and stress_converged

        return SCFTState(
            step=step,
            elapsed=elapsed,
            is_converged=is_converged,
            field_error=field_error,
            stress_error=stress_error,
            free_energy=self.system.free_energy(),
            free_energy_bulk=self.system.free_energy_bulk(),
        )
