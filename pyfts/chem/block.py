from dataclasses import dataclass

from pyfts.chem.monomer import Monomer


@dataclass
class Block:
    monomer: Monomer
    repeat_units: int
    segment_length: float

    @property
    def size(self) -> float:
        return self.repeat_units * self.monomer.size
