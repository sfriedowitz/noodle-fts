import pytest
from pydantic import ValidationError

from pynoodle import System, SystemBuilder
from pynoodle.configs import (
    BlockConfig,
    InteractionConfig,
    LamellarCellConfig,
    MeshConfig,
    MonomerConfig,
    PolymerConfig,
    SystemConfig,
)


class TestInteractionConfig:
    def test_valid_config(self):
        config = InteractionConfig(i=0, j=1, chi=0.5)
        assert config.i == 0
        assert config.j == 1
        assert config.chi == 0.5

    def test_negative_ids(self):
        with pytest.raises(ValidationError):
            InteractionConfig(i=-1, j=0, chi=0.5)


class TestSystemConfig:
    @pytest.fixture
    def diblock_config(self):
        monomer_a = MonomerConfig(id=0, volume=1.0)
        monomer_b = MonomerConfig(id=1, volume=1.0)
        block_a = BlockConfig(monomer_id=0, repeat_units=10, segment_length=0.1)
        block_b = BlockConfig(monomer_id=1, repeat_units=10, segment_length=0.1)

        return SystemConfig(
            monomers=[monomer_a, monomer_b],
            mesh=MeshConfig(dimensions=[32]),
            cell=LamellarCellConfig(a=4.0),
            species=[PolymerConfig(blocks=[block_a, block_b], contour_steps=100, phi=1.0)],
            interactions=[InteractionConfig(i=0, j=1, chi=0.5)],
        )

    def test_valid_config(self, diblock_config):
        assert len(diblock_config.species) == 1
        assert len(diblock_config.interactions) == 1

    def test_invalid_volume_fractions(self):
        monomer_a = MonomerConfig(id=0, volume=1.0)
        block_a = BlockConfig(monomer_id=0, repeat_units=10, segment_length=0.1)

        with pytest.raises(ValidationError, match="volume fractions"):
            SystemConfig(
                monomers=[monomer_a],
                mesh=MeshConfig(dimensions=[32]),
                cell=LamellarCellConfig(a=4.0),
                species=[PolymerConfig(blocks=[block_a], contour_steps=100, phi=0.5)],
            )

    def test_build(self, diblock_config):
        system = SystemBuilder.build_system(diblock_config)
        assert isinstance(system, System)
        assert system.nmonomer == 2
        assert system.nspecies == 1

    def test_yaml_roundtrip(self, diblock_config, tmp_path):
        yaml_str = diblock_config.to_yaml()
        assert "mesh:" in yaml_str
        assert "cell:" in yaml_str
        assert "species:" in yaml_str

        config_from_yaml = SystemConfig.from_yaml(yaml_str)
        assert len(config_from_yaml.species) == 1
        assert config_from_yaml.mesh.dimensions == [32]

    def test_yaml_file_io(self, diblock_config, tmp_path):
        yaml_file = tmp_path / "system.yaml"

        diblock_config.to_yaml_file(yaml_file)
        assert yaml_file.exists()

        config_from_file = SystemConfig.from_yaml_file(yaml_file)
        assert len(config_from_file.species) == 1
        assert config_from_file.cell.a == 4.0

    def test_multiple_species(self):
        monomer_a = MonomerConfig(id=0, volume=1.0)
        monomer_b = MonomerConfig(id=1, volume=1.0)
        block_a = BlockConfig(monomer_id=0, repeat_units=10, segment_length=0.1)
        block_b = BlockConfig(monomer_id=1, repeat_units=10, segment_length=0.1)

        config = SystemConfig(
            monomers=[monomer_a, monomer_b],
            mesh=MeshConfig(dimensions=[32]),
            cell=LamellarCellConfig(a=4.0),
            species=[
                PolymerConfig(blocks=[block_a, block_b], contour_steps=100, phi=0.6),
                PolymerConfig(blocks=[block_b, block_a], contour_steps=100, phi=0.4),
            ],
        )

        system = SystemBuilder.build_system(config)
        assert system.nspecies == 2
