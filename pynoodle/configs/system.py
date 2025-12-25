from pydantic import Field, model_validator

from pynoodle.configs.base import BaseNoodleModel
from pynoodle.configs.chem import MonomerConfig, PointConfig, PolymerConfig, SpeciesConfig
from pynoodle.configs.domain import MeshConfig, UnitCellConfig


class InteractionConfig(BaseNoodleModel):
    """Configuration for Flory-Huggins interaction parameters."""

    i: int = Field(ge=0)
    j: int = Field(ge=0)
    chi: float


class SystemConfig(BaseNoodleModel):
    """Configuration for a polymer field-theoretic system."""

    mesh: MeshConfig
    cell: UnitCellConfig
    monomers: list[MonomerConfig] = Field(min_length=1)
    species: list[SpeciesConfig] = Field(min_length=1)
    interactions: list[InteractionConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_system(self) -> "SystemConfig":
        """Validate that volume fractions sum to ~1.0 and monomer references are valid."""
        # Check volume fractions
        total_phi = sum(s.phi for s in self.species)
        if not 0.99 <= total_phi <= 1.01:
            msg = f"Species volume fractions must sum to 1.0, got {total_phi}"
            raise ValueError(msg)

        # Check for duplicate monomer IDs
        defined_ids = [m.id for m in self.monomers]
        if len(defined_ids) != len(set(defined_ids)):
            msg = f"Duplicate monomer IDs found: {defined_ids}"
            raise ValueError(msg)

        # Convert to set for lookup
        defined_ids_set = set(defined_ids)

        # Get all monomer IDs referenced by species
        referenced_ids = set()
        for species in self.species:
            if isinstance(species, PointConfig):
                referenced_ids.add(species.monomer_id)
            elif isinstance(species, PolymerConfig):
                for block in species.blocks:
                    referenced_ids.add(block.monomer_id)

        # Check that all referenced monomer IDs exist
        missing_ids = referenced_ids - defined_ids_set
        if missing_ids:
            msg = f"Species reference undefined monomer IDs: {sorted(missing_ids)}"
            raise ValueError(msg)

        # Check that interaction IDs reference defined monomers
        for interaction in self.interactions:
            if interaction.i not in defined_ids_set:
                msg = f"Interaction references undefined monomer ID: {interaction.i}"
                raise ValueError(msg)
            if interaction.j not in defined_ids_set:
                msg = f"Interaction references undefined monomer ID: {interaction.j}"
                raise ValueError(msg)

        return self
