import numpy as np
import pytest

from pynoodle import (
    Block,
    CubicCell,
    Hexagonal2DCell,
    LamellarCell,
    Mesh,
    Monomer,
    Point,
    Polymer,
    SquareCell,
    System,
)


class TestMonomer:
    def test_creation(self) -> None:
        m = Monomer(id=0, volume=1.0)
        assert m.id == 0
        assert m.volume == 1.0

    def test_repr(self, monomer_a: Monomer) -> None:
        assert "Monomer" in repr(monomer_a)
        assert "id=0" in repr(monomer_a)


class TestBlock:
    def test_creation(self, monomer_a: Monomer) -> None:
        b = Block(monomer=monomer_a, repeat_units=10, segment_length=0.1)
        assert b.monomer.id == monomer_a.id
        assert b.repeat_units == 10
        assert b.segment_length == 0.1


class TestPoint:
    def test_creation(self, monomer_a: Monomer) -> None:
        p = Point(monomer=monomer_a, phi=1.0)
        assert p.phi == 1.0
        assert p.volume == monomer_a.volume
        assert len(p.monomers) == 1
        assert p.monomer_fraction(0) == 1.0


class TestPolymer:
    def test_creation(self, block_a: Block, block_b: Block) -> None:
        poly = Polymer(blocks=[block_a, block_b], contour_steps=100, phi=1.0)
        assert poly.phi == 1.0
        assert len(poly.blocks) == 2
        assert len(poly.monomers) == 2

    def test_monomer_fraction(self, diblock_polymer: Polymer) -> None:
        frac_a = diblock_polymer.monomer_fraction(0)
        frac_b = diblock_polymer.monomer_fraction(1)
        assert abs(frac_a - 0.5) < 1e-6
        assert abs(frac_b - 0.5) < 1e-6


class TestMesh:
    def test_1d_mesh(self, mesh_1d: Mesh) -> None:
        assert mesh_1d.ndim == 1
        assert mesh_1d.size == 32
        assert mesh_1d.dimensions == [32]

    def test_2d_mesh(self, mesh_2d: Mesh) -> None:
        assert mesh_2d.ndim == 2
        assert mesh_2d.size == 32 * 32
        assert mesh_2d.dimensions == [32, 32]

    def test_3d_mesh(self, mesh_3d: Mesh) -> None:
        assert mesh_3d.ndim == 3
        assert mesh_3d.size == 16 * 16 * 16
        assert mesh_3d.dimensions == [16, 16, 16]


class TestUnitCell:
    def test_lamellar_cell(self, lamellar_cell: LamellarCell) -> None:
        assert lamellar_cell.ndim == 1
        assert lamellar_cell.volume > 0
        assert len(lamellar_cell.parameters) == 1
        assert lamellar_cell.parameters[0] == 4.0

    def test_square_cell(self) -> None:
        cell = SquareCell(a=4.0)
        assert cell.ndim == 2
        assert cell.volume > 0
        assert len(cell.parameters) == 1

    def test_hexagonal2d_cell(self) -> None:
        cell = Hexagonal2DCell(a=4.0)
        assert cell.ndim == 2
        assert cell.volume > 0

    def test_cubic_cell(self) -> None:
        cell = CubicCell(a=4.0)
        assert cell.ndim == 3
        assert cell.volume == pytest.approx(4.0**3)


class TestSystem:
    def test_creation(
        self, mesh_1d: Mesh, lamellar_cell: LamellarCell, diblock_polymer: Polymer
    ) -> None:
        system = System(mesh=mesh_1d, cell=lamellar_cell, species=[diblock_polymer])
        assert system.nmonomer == 2
        assert system.nspecies == 1
        assert system.mesh.ndim == 1
        assert system.cell.ndim == 1

    def test_fields_and_concentrations(
        self,
        mesh_1d: Mesh,
        lamellar_cell: LamellarCell,
        diblock_polymer: Polymer,
    ) -> None:
        system = System(mesh=mesh_1d, cell=lamellar_cell, species=[diblock_polymer])

        fields = system.get_fields()
        concentrations = system.get_concentrations()

        assert len(fields) == 2
        assert len(concentrations) == 2

        for field in fields.values():
            assert field.shape == (32,)

    def test_set_interaction(
        self,
        mesh_1d: Mesh,
        lamellar_cell: LamellarCell,
        diblock_polymer: Polymer,
    ) -> None:
        system = System(mesh=mesh_1d, cell=lamellar_cell, species=[diblock_polymer])
        system.set_interaction(0, 1, 0.5)

    def test_sample_fields(
        self,
        mesh_1d: Mesh,
        lamellar_cell: LamellarCell,
        diblock_polymer: Polymer,
    ) -> None:
        system = System(mesh=mesh_1d, cell=lamellar_cell, species=[diblock_polymer])

        system.sample_fields(scale=0.1, seed=42)

        fields = system.get_fields()
        for field in fields.values():
            assert not np.allclose(field, 0.0)
