from typing import Literal

from pydantic import Field, PositiveFloat, PositiveInt

from pynoodle.configs.base import BaseNoodleModel


class MeshConfig(BaseNoodleModel):
    """Configuration for spatial discretization mesh."""

    dimensions: list[PositiveInt] = Field(min_length=1, max_length=3)


class LamellarCellConfig(BaseNoodleModel):
    """Configuration for 1D lamellar unit cell."""

    type: Literal["lamellar"] = "lamellar"
    a: PositiveFloat


class SquareCellConfig(BaseNoodleModel):
    """Configuration for 2D square unit cell."""

    type: Literal["square"] = "square"
    a: PositiveFloat


class RectangularCellConfig(BaseNoodleModel):
    """Configuration for 2D rectangular unit cell."""

    type: Literal["rectangular"] = "rectangular"
    a: PositiveFloat
    b: PositiveFloat


class Hexagonal2DCellConfig(BaseNoodleModel):
    """Configuration for 2D hexagonal unit cell."""

    type: Literal["hexagonal2d"] = "hexagonal2d"
    a: PositiveFloat


class ObliqueCellConfig(BaseNoodleModel):
    """Configuration for 2D oblique unit cell."""

    type: Literal["oblique"] = "oblique"
    a: PositiveFloat
    b: PositiveFloat
    gamma: PositiveFloat


class CubicCellConfig(BaseNoodleModel):
    """Configuration for 3D cubic unit cell."""

    type: Literal["cubic"] = "cubic"
    a: PositiveFloat


class TetragonalCellConfig(BaseNoodleModel):
    """Configuration for 3D tetragonal unit cell."""

    type: Literal["tetragonal"] = "tetragonal"
    a: PositiveFloat
    c: PositiveFloat


class OrthorhombicCellConfig(BaseNoodleModel):
    """Configuration for 3D orthorhombic unit cell."""

    type: Literal["orthorhombic"] = "orthorhombic"
    a: PositiveFloat
    b: PositiveFloat
    c: PositiveFloat


class RhombohedralCellConfig(BaseNoodleModel):
    """Configuration for 3D rhombohedral unit cell."""

    type: Literal["rhombohedral"] = "rhombohedral"
    a: PositiveFloat
    alpha: PositiveFloat


class Hexagonal3DCellConfig(BaseNoodleModel):
    """Configuration for 3D hexagonal unit cell."""

    type: Literal["hexagonal3d"] = "hexagonal3d"
    a: PositiveFloat
    c: PositiveFloat


class MonoclinicCellConfig(BaseNoodleModel):
    """Configuration for 3D monoclinic unit cell."""

    type: Literal["monoclinic"] = "monoclinic"
    a: PositiveFloat
    b: PositiveFloat
    c: PositiveFloat
    beta: PositiveFloat


class TriclinicCellConfig(BaseNoodleModel):
    """Configuration for 3D triclinic unit cell."""

    type: Literal["triclinic"] = "triclinic"
    a: PositiveFloat
    b: PositiveFloat
    c: PositiveFloat
    alpha: PositiveFloat
    beta: PositiveFloat
    gamma: PositiveFloat


# Union type for unit cell configurations
UnitCellConfig = (
    LamellarCellConfig
    | SquareCellConfig
    | RectangularCellConfig
    | Hexagonal2DCellConfig
    | ObliqueCellConfig
    | CubicCellConfig
    | TetragonalCellConfig
    | OrthorhombicCellConfig
    | RhombohedralCellConfig
    | Hexagonal3DCellConfig
    | MonoclinicCellConfig
    | TriclinicCellConfig
)
