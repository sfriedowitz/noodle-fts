from collections.abc import Iterable

import numpy as np

from pyfts.domain.mesh import Mesh
from pyfts.system.system import System
from pyfts.updaters.base import FieldUpdater


class EulerUpdater(FieldUpdater):
    def __init__(self, mesh: Mesh, monomer_ids: Iterable[str], dt: float):
        self.dt = dt
        self._buffers = {id: np.zeros(mesh.dimensions) for id in monomer_ids}

    @classmethod
    def from_system(cls, system: System, dt: float) -> "EulerUpdater":
        return EulerUpdater(system.mesh, system.monomers().keys(), dt)

    def step(self, system: System) -> None:
        return None
