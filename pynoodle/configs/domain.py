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


class Hexagonal2DCellConfig(BaseNoodleModel):
    """Configuration for 2D hexagonal unit cell."""

    type: Literal["hexagonal2d"] = "hexagonal2d"
    a: PositiveFloat


class CubicCellConfig(BaseNoodleModel):
    """Configuration for 3D cubic unit cell."""

    type: Literal["cubic"] = "cubic"
    a: PositiveFloat


# Union type for unit cell configurations
UnitCellConfig = LamellarCellConfig | SquareCellConfig | Hexagonal2DCellConfig | CubicCellConfig
