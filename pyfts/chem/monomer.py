from dataclasses import dataclass


@dataclass
class Monomer:
    """Description of a chemical monomer.

    The `size` is the ratio of the monomer volume to some arbitrary reference volume.
    """

    id: str
    size: float
