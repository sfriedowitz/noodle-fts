import numpy as np

from pyfts.chem.monomer import Monomer
from pyfts.chem.species import Species
from pyfts.domain.cell import UnitCell
from pyfts.domain.mesh import Mesh
from pyfts.solvers import build_solver


class System:
    def __init__(self, mesh: Mesh, cell: UnitCell, species: list[Species]):
        if mesh.ndim != cell.ndim:
            raise ValueError("mesh dimension != cell dimension")

        self.mesh = mesh
        self.cell = cell
        self.species = species
        self._solvers = [build_solver(s) for s in self.species]

        # State
        monomer_ids = set(self.monomers().keys())
        self.ksq = np.zeros(self.mesh.kmesh())
        self.fields = {id: np.zeros(self.mesh) for id in monomer_ids}
        self.concentrations = {id: np.zeros(self.mesh) for id in monomer_ids}
        self.residuals = {id: np.zeros(self.mesh) for id in monomer_ids}
        self.total_concentration = np.zeros(self.mesh)

    def monomers(self) -> dict[str, Monomer]:
        return {id: m for s in self.species for id, m in s.monomers().items()}

    def solve(self) -> None:
        pass
