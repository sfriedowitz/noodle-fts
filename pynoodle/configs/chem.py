"""Pydantic configuration models for chemical components."""

from typing import Annotated, Literal

from pydantic import Field, NonNegativeInt, PositiveFloat, PositiveInt

from pynoodle.configs.base import BaseNoodleModel

VolumeFraction = Annotated[float, Field(gt=0.0, le=1.0)]


class MonomerConfig(BaseNoodleModel):
    """Configuration for a monomer type."""

    id: int = Field(ge=0)
    volume: PositiveFloat


class BlockConfig(BaseNoodleModel):
    """Configuration for a polymer block."""

    monomer_id: NonNegativeInt
    repeat_units: PositiveInt
    segment_length: PositiveFloat


class PointConfig(BaseNoodleModel):
    """Configuration for a point-like species."""

    type: Literal["point"] = "point"
    monomer_id: NonNegativeInt
    phi: VolumeFraction


class PolymerConfig(BaseNoodleModel):
    """Configuration for a polymeric species."""

    type: Literal["polymer"] = "polymer"
    blocks: list[BlockConfig] = Field(min_length=1)
    contour_steps: PositiveInt
    phi: VolumeFraction


SpeciesConfig = Annotated[PointConfig | PolymerConfig, Field(discriminator="type")]
