"""Pydantic configuration models for chemical components."""

from typing import Annotated, Literal

from pydantic import Field

from pynoodle.configs.base import BaseNoodleModel


class MonomerConfig(BaseNoodleModel):
    """Configuration for a monomer type."""

    id: int = Field(ge=0)
    size: float = Field(gt=0.0)


class BlockConfig(BaseNoodleModel):
    """Configuration for a polymer block."""

    monomer_id: int = Field(ge=0)
    repeat_units: int = Field(gt=0)
    segment_length: float = Field(gt=0.0)


class PointConfig(BaseNoodleModel):
    """Configuration for a point-like species."""

    type: Literal["point"] = "point"
    monomer_id: int = Field(ge=0)
    phi: float = Field(gt=0.0, le=1.0)


class PolymerConfig(BaseNoodleModel):
    """Configuration for a polymeric species."""

    type: Literal["polymer"] = "polymer"
    blocks: list[BlockConfig] = Field(min_length=1)
    contour_steps: int = Field(gt=0)
    phi: float = Field(gt=0.0, le=1.0)


SpeciesConfig = Annotated[PointConfig | PolymerConfig, Field(discriminator="type")]
