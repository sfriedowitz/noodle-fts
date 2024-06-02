import datetime
from dataclasses import dataclass


@dataclass
class SCFTConfig:
    steps: int
    step_size: float
    field_tolerance: float


@dataclass
class SCFTState:
    step: float
    elapsed: datetime.timedelta
    is_converged: bool
    field_error: float
    free_energy: float
    free_energy_bulk: float


def scft(*, config: SCFTConfig | None = None) -> SCFTState:
    pass
