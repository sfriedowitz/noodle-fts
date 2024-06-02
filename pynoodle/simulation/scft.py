from dataclasses import dataclass
from datetime import datetime, timedelta

from tqdm import tqdm

from pynoodle import FieldUpdater, System


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
    field_tolerance: float | None = None,
) -> SCFTState:
    start = datetime.now()
    pbar = tqdm(range(steps), desc="SCFT")
    for step in pbar:
        state = _get_state(system, start, step, field_tolerance)
        pbar.set_postfix(f=state.free_energy, err=state.field_error)
        if state.is_converged:
            return state
        updater.step(system)
    return _get_state(system, start, steps, field_tolerance)


def _get_state(
    system: System,
    start: datetime,
    step: int,
    field_tolerance: float | None,
) -> SCFTState:
    elapsed = datetime.now() - start
    field_error = system.field_error()
    is_converged = field_error <= field_tolerance if field_tolerance else False
    return SCFTState(
        step=step,
        elapsed=elapsed,
        is_converged=is_converged,
        field_error=field_error,
        free_energy=system.free_energy(),
        free_energy_bulk=system.free_energy_bulk(),
    )
