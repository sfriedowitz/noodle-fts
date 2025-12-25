import numpy as np
from numpy.typing import NDArray

class Monomer:
    """A monomer type with an ID and statistical segment size."""

    def __init__(self, id: int, size: float) -> None: ...
    @property
    def id(self) -> int:
        """Unique identifier for this monomer type."""
        ...
    @property
    def size(self) -> float:
        """Statistical segment size (Kuhn length)."""
        ...

class Block:
    """A polymer block composed of repeated monomer units."""

    def __init__(self, monomer: Monomer, repeat_units: int, segment_length: float) -> None: ...
    @property
    def monomer(self) -> Monomer:
        """The monomer type for this block."""
        ...
    @property
    def repeat_units(self) -> int:
        """Number of monomer repeat units."""
        ...
    @property
    def segment_length(self) -> float:
        """Length of each segment along the polymer contour."""
        ...

class Species:
    """Base class for molecular species (Point or Polymer)."""

    @property
    def phi(self) -> float:
        """Volume fraction of this species."""
        ...
    @property
    def size(self) -> float:
        """Total size of this species."""
        ...
    @property
    def monomers(self) -> list[Monomer]:
        """Return a list of monomers in this species."""
        ...
    def get_monomer_fraction(self, id: int) -> float:
        """Get the fraction of this species composed of monomer with given id."""
        ...

class Point(Species):
    """A point-like species (non-polymeric)."""

    def __init__(self, monomer: Monomer, phi: float) -> None: ...

class Polymer(Species):
    """A polymeric species composed of one or more blocks."""

    def __init__(self, blocks: list[Block], contour_steps: int, phi: float) -> None: ...
    @property
    def blocks(self) -> list[Block]:
        """List of polymer blocks."""
        ...

class Mesh:
    """Spatial discretization mesh."""

    def __init__(self, *dimensions: int) -> None: ...
    @property
    def ndim(self) -> int:
        """Number of spatial dimensions."""
        ...
    @property
    def size(self) -> int:
        """Total number of grid points."""
        ...
    @property
    def dimensions(self) -> list[int]:
        """Grid dimensions in each direction."""
        ...

class UnitCell:
    """Base class for unit cell geometries."""

    @property
    def ndim(self) -> int:
        """Number of spatial dimensions."""
        ...
    @property
    def volume(self) -> float:
        """Unit cell volume."""
        ...
    @property
    def parameters(self) -> NDArray[np.floating]:
        """Cell parameters (lattice constants)."""
        ...
    @property
    def shape(self) -> NDArray[np.floating]:
        """Shape matrix defining unit cell geometry."""
        ...
    @property
    def metric(self) -> NDArray[np.floating]:
        """Metric tensor."""
        ...

class LamellarCell(UnitCell):
    """1D lamellar unit cell."""

    def __init__(self, a: float) -> None: ...

class SquareCell(UnitCell):
    """2D square unit cell."""

    def __init__(self, a: float) -> None: ...

class RectangularCell(UnitCell):
    """2D rectangular unit cell."""

    def __init__(self, a: float, b: float) -> None: ...

class Hexagonal2DCell(UnitCell):
    """2D hexagonal unit cell."""

    def __init__(self, a: float) -> None: ...

class ObliqueCell(UnitCell):
    """2D oblique unit cell."""

    def __init__(self, a: float, b: float, gamma: float) -> None: ...

class CubicCell(UnitCell):
    """3D cubic unit cell."""

    def __init__(self, a: float) -> None: ...

class TetragonalCell(UnitCell):
    """3D tetragonal unit cell."""

    def __init__(self, a: float, c: float) -> None: ...

class OrthorhombicCell(UnitCell):
    """3D orthorhombic unit cell."""

    def __init__(self, a: float, b: float, c: float) -> None: ...

class RhombohedralCell(UnitCell):
    """3D rhombohedral unit cell."""

    def __init__(self, a: float, alpha: float) -> None: ...

class Hexagonal3DCell(UnitCell):
    """3D hexagonal unit cell."""

    def __init__(self, a: float, c: float) -> None: ...

class MonoclinicCell(UnitCell):
    """3D monoclinic unit cell."""

    def __init__(self, a: float, b: float, c: float, beta: float) -> None: ...

class TriclinicCell(UnitCell):
    """3D triclinic unit cell."""

    def __init__(
        self,
        a: float,
        b: float,
        c: float,
        alpha: float,
        beta: float,
        gamma: float,
    ) -> None: ...

class System:
    """Polymer field-theoretic system."""

    def __init__(self, mesh: Mesh, cell: UnitCell, species: list[Species]) -> None: ...
    @property
    def nmonomer(self) -> int:
        """Number of distinct monomer types."""
        ...
    @property
    def nspecies(self) -> int:
        """Number of molecular species."""
        ...
    @property
    def mesh(self) -> Mesh:
        """Spatial discretization mesh."""
        ...
    @property
    def cell(self) -> UnitCell:
        """Unit cell geometry."""
        ...
    def get_free_energy(self) -> float:
        """Calculate total free energy."""
        ...
    def get_free_energy_bulk(self) -> float:
        """Calculate bulk free energy."""
        ...
    def get_field_error(self) -> float:
        """Calculate field error (L2 norm of field residuals)."""
        ...
    def get_stress_error(self) -> float:
        """Calculate stress error (Frobenius norm of stress tensor)."""
        ...
    def set_interaction(self, i: int, j: int, chi: float) -> None:
        """Set Flory-Huggins interaction parameter."""
        ...
    def get_fields(self) -> dict[int, NDArray[np.floating]]:
        """Get all fields indexed by monomer ID."""
        ...
    def get_concentrations(self) -> dict[int, NDArray[np.floating]]:
        """Get all concentration fields indexed by monomer ID."""
        ...
    def get_stress(self) -> NDArray[np.floating]:
        """Get stress tensor."""
        ...
    def set_field(self, id: int, field: NDArray[np.floating]) -> None:
        """Set field for a specific monomer."""
        ...
    def set_concentration(self, id: int, concentration: NDArray[np.floating]) -> None:
        """Set concentration field for a specific monomer."""
        ...
    def sample_fields(self, *, scale: float = 0.01, seed: int | None = None) -> None:
        """Sample fields from normal distribution."""
        ...

class FieldUpdater:
    """Updates fields using SCFT iteration."""

    def __init__(self, system: System, step_size: float) -> None: ...
    @property
    def step_size(self) -> float:
        """Step size for field updates."""
        ...
    def step(self, system: System) -> None:
        """Perform one field update step."""
        ...

class CellUpdater:
    """Updates unit cell parameters based on stress."""

    def __init__(self, step_size: float) -> None: ...
    @property
    def step_size(self) -> float:
        """Step size for cell parameter updates."""
        ...
    def step(self, system: System) -> None:
        """Perform one cell update step."""
        ...
