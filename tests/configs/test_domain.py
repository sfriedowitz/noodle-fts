import pytest
from pydantic import ValidationError

from pynoodle.configs import (
    CubicCellConfig,
    Hexagonal2DCellConfig,
    LamellarCellConfig,
    MeshConfig,
    SquareCellConfig,
)


class TestMeshConfig:
    def test_valid_1d_config(self):
        config = MeshConfig(dimensions=[32])
        assert config.dimensions == [32]

    def test_valid_2d_config(self):
        config = MeshConfig(dimensions=[32, 32])
        assert config.dimensions == [32, 32]

    def test_valid_3d_config(self):
        config = MeshConfig(dimensions=[16, 16, 16])
        assert config.dimensions == [16, 16, 16]

    def test_invalid_dimensions(self):
        with pytest.raises(ValidationError):
            MeshConfig(dimensions=[])
        with pytest.raises(ValidationError):
            MeshConfig(dimensions=[32, 32, 32, 32])

    def test_negative_dimensions(self):
        with pytest.raises(ValidationError):
            MeshConfig(dimensions=[-32])
        with pytest.raises(ValidationError):
            MeshConfig(dimensions=[32, 0])


class TestLamellarCellConfig:
    def test_valid_config(self):
        config = LamellarCellConfig(a=4.0)
        assert config.a == 4.0
        assert config.type == "lamellar"

    def test_invalid_lattice_constant(self):
        with pytest.raises(ValidationError):
            LamellarCellConfig(a=0.0)
        with pytest.raises(ValidationError):
            LamellarCellConfig(a=-1.0)


class TestSquareCellConfig:
    def test_valid_config(self):
        config = SquareCellConfig(a=4.0)
        assert config.type == "square"


class TestHexagonal2DCellConfig:
    def test_valid_config(self):
        config = Hexagonal2DCellConfig(a=4.0)
        assert config.type == "hexagonal2d"


class TestCubicCellConfig:
    def test_valid_config(self):
        config = CubicCellConfig(a=4.0)
        assert config.type == "cubic"
