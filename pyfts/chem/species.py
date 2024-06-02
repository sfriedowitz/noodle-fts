from abc import ABC, abstractmethod
from dataclasses import dataclass

from pyfts.chem.block import Block
from pyfts.chem.monomer import Monomer


@dataclass
class Species(ABC):
    phi: float

    @property
    @abstractmethod
    def size(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def monomers(self) -> dict[str, Monomer]:
        raise NotImplementedError

    @abstractmethod
    def monomer_fraction(self, id: str) -> float:
        raise NotImplementedError


@dataclass
class Point(Species):
    monomer: Monomer

    @property
    def size(self) -> float:
        return self.monomer.size

    def monomers(self) -> dict[str, Monomer]:
        return {self.monomer.id: self.monomer}

    def monomer_fraction(self, id: str) -> float:
        return float(id == self.monomer.id)


@dataclass
class Polymer(Species):
    blocks: list[Block]
    contour_steps: int

    @property
    def size(self) -> float:
        return sum(b.size for b in self.blocks)

    @property
    def nblock(self) -> int:
        return len(self.blocks)

    def monomers(self) -> dict[str, Monomer]:
        return {b.monomer.id: b.monomer for b in self.blocks}

    def monomer_fraction(self, id: str) -> float:
        blocks_with_monomer = sum(id == b.monomer.id for b in self.blocks)
        return blocks_with_monomer / self.nblock
