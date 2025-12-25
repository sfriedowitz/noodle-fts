import pytest
from pydantic import ValidationError

from pynoodle.configs import (
    CubicCellConfig,
    Hexagonal2DCellConfig,
    Hexagonal3DCellConfig,
    LamellarCellConfig,
    MeshConfig,
    MonoclinicCellConfig,
    ObliqueCellConfig,
    OrthorhombicCellConfig,
    RectangularCellConfig,
    RhombohedralCellConfig,
    SquareCellConfig,
    TetragonalCellConfig,
    TriclinicCellConfig,
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


class TestRectangularCellConfig:
    def test_valid_config(self):
        config = RectangularCellConfig(a=4.0, b=3.0)
        assert config.type == "rectangular"
        assert config.a == 4.0
        assert config.b == 3.0


class TestHexagonal2DCellConfig:
    def test_valid_config(self):
        config = Hexagonal2DCellConfig(a=4.0)
        assert config.type == "hexagonal2d"


class TestObliqueCellConfig:
    def test_valid_config(self):
        config = ObliqueCellConfig(a=4.0, b=3.0, gamma=1.047)
        assert config.type == "oblique"
        assert config.a == 4.0
        assert config.b == 3.0
        assert config.gamma == 1.047


class TestCubicCellConfig:
    def test_valid_config(self):
        config = CubicCellConfig(a=4.0)
        assert config.type == "cubic"


class TestTetragonalCellConfig:
    def test_valid_config(self):
        config = TetragonalCellConfig(a=4.0, c=5.0)
        assert config.type == "tetragonal"
        assert config.a == 4.0
        assert config.c == 5.0


class TestOrthorhombicCellConfig:
    def test_valid_config(self):
        config = OrthorhombicCellConfig(a=4.0, b=3.0, c=5.0)
        assert config.type == "orthorhombic"
        assert config.a == 4.0
        assert config.b == 3.0
        assert config.c == 5.0


class TestRhombohedralCellConfig:
    def test_valid_config(self):
        config = RhombohedralCellConfig(a=4.0, alpha=1.571)
        assert config.type == "rhombohedral"
        assert config.a == 4.0
        assert config.alpha == 1.571


class TestHexagonal3DCellConfig:
    def test_valid_config(self):
        config = Hexagonal3DCellConfig(a=4.0, c=5.0)
        assert config.type == "hexagonal3d"
        assert config.a == 4.0
        assert config.c == 5.0


class TestMonoclinicCellConfig:
    def test_valid_config(self):
        config = MonoclinicCellConfig(a=4.0, b=3.0, c=5.0, beta=1.571)
        assert config.type == "monoclinic"
        assert config.a == 4.0
        assert config.b == 3.0
        assert config.c == 5.0
        assert config.beta == 1.571


class TestTriclinicCellConfig:
    def test_valid_config(self):
        config = TriclinicCellConfig(a=4.0, b=3.0, c=5.0, alpha=1.571, beta=1.571, gamma=1.047)
        assert config.type == "triclinic"
        assert config.a == 4.0
        assert config.b == 3.0
        assert config.c == 5.0
        assert config.alpha == 1.571
        assert config.beta == 1.571
        assert config.gamma == 1.047
