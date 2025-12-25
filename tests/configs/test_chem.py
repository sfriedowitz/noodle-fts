import pytest
from pydantic import ValidationError

from pynoodle.configs import BlockConfig, MonomerConfig, PointConfig, PolymerConfig


class TestMonomerConfig:
    def test_valid_config(self):
        config = MonomerConfig(id=0, volume=1.0)
        assert config.id == 0
        assert config.volume == 1.0

    def test_invalid_id(self):
        with pytest.raises(ValidationError):
            MonomerConfig(id=-1, volume=1.0)

    def test_invalid_volume(self):
        with pytest.raises(ValidationError):
            MonomerConfig(id=0, volume=0.0)
        with pytest.raises(ValidationError):
            MonomerConfig(id=0, volume=-1.0)


class TestBlockConfig:
    def test_valid_config(self):
        config = BlockConfig(monomer_id=0, repeat_units=10, segment_length=0.1)
        assert config.monomer_id == 0
        assert config.repeat_units == 10
        assert config.segment_length == 0.1

    def test_invalid_repeat_units(self):
        with pytest.raises(ValidationError):
            BlockConfig(monomer_id=0, repeat_units=0, segment_length=0.1)


class TestPointConfig:
    def test_valid_config(self):
        config = PointConfig(monomer_id=0, phi=0.5)
        assert config.monomer_id == 0
        assert config.phi == 0.5
        assert config.type == "point"

    def test_invalid_phi(self):
        with pytest.raises(ValidationError):
            PointConfig(monomer_id=0, phi=0.0)
        with pytest.raises(ValidationError):
            PointConfig(monomer_id=0, phi=1.5)


class TestPolymerConfig:
    def test_valid_config(self):
        block_a = BlockConfig(monomer_id=0, repeat_units=10, segment_length=0.1)
        block_b = BlockConfig(monomer_id=1, repeat_units=10, segment_length=0.1)
        config = PolymerConfig(blocks=[block_a, block_b], contour_steps=100, phi=1.0)
        assert len(config.blocks) == 2
        assert config.contour_steps == 100
        assert config.type == "polymer"

    def test_empty_blocks(self):
        with pytest.raises(ValidationError):
            PolymerConfig(blocks=[], contour_steps=100, phi=1.0)
