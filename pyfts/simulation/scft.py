from dataclasses import dataclass
from datetime import datetime, timedelta

from pyfts import FieldUpdater, System


@dataclass
class SCFTState:
    step: float
    elapsed: timedelta
    is_converged: bool
    field_error: float
    free_energy: float
    free_energy_bulk: float


def scft(
    system: System,
    updater: FieldUpdater,
    *,
    steps: int = 100,
    field_tolerance: float = 1e-5,
) -> SCFTState:
    start = datetime.now()
    for step in range(steps):
        state = _get_state(system, start, step, field_tolerance)
        if state.is_converged:
            return state
        updater.step(system)

    return _get_state(system, start, steps, field_tolerance)


def _get_state(system: System, start: datetime, step: int, field_tolerance: float) -> SCFTState:
    elapsed = datetime.now() - start
    field_error = system.field_error()
    is_converged = field_error <= field_tolerance
    return SCFTState(
        step=step,
        elapsed=elapsed,
        is_converged=is_converged,
        field_error=field_error,
        free_energy=system.free_energy(),
        free_energy_bulk=system.free_energy_bulk(),
    )
