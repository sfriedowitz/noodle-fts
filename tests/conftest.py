import pytest

from pynoodle import Block, LamellarCell, Mesh, Monomer, Polymer
from pynoodle.configs import MonomerConfig


@pytest.fixture
def monomer_a():
    """Create a test monomer A."""
    return Monomer(id=0, volume=1.0)


@pytest.fixture
def monomer_b():
    """Create a test monomer B."""
    return Monomer(id=1, volume=1.0)


@pytest.fixture
def monomer_a_config():
    """Create a test monomer A configuration."""
    return MonomerConfig(id=0, volume=1.0)


@pytest.fixture
def monomer_b_config():
    """Create a test monomer B configuration."""
    return MonomerConfig(id=1, volume=1.0)


@pytest.fixture
def block_a(monomer_a):
    """Create a test block of monomer A."""
    return Block(monomer=monomer_a, repeat_units=10, segment_length=0.1)


@pytest.fixture
def block_b(monomer_b):
    """Create a test block of monomer B."""
    return Block(monomer=monomer_b, repeat_units=10, segment_length=0.1)


@pytest.fixture
def diblock_polymer(block_a, block_b):
    """Create a test AB diblock copolymer."""
    return Polymer(blocks=[block_a, block_b], contour_steps=100, phi=1.0)


@pytest.fixture
def mesh_1d():
    """Create a 1D test mesh."""
    return Mesh(32)


@pytest.fixture
def mesh_2d():
    """Create a 2D test mesh."""
    return Mesh(32, 32)


@pytest.fixture
def mesh_3d():
    """Create a 3D test mesh."""
    return Mesh(16, 16, 16)


@pytest.fixture
def lamellar_cell():
    """Create a test lamellar cell."""
    return LamellarCell(a=4.0)
