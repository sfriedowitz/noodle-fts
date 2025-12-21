"""Integration tests for variable cell SCFT optimization."""

import pytest

from pynoodle import Block, CellUpdater, FieldUpdater, LamellarCell, Mesh, Monomer, Polymer, System


@pytest.fixture
def diblock_system():
    """Create a diblock copolymer system for testing."""
    mesh = Mesh.One(32)
    cell = LamellarCell(10.0)

    monomer_a = Monomer(0, 1.0)
    monomer_b = Monomer(1, 1.0)

    block_a = Block(monomer_a, 50, 1.0)
    block_b = Block(monomer_b, 50, 1.0)

    polymer = Polymer([block_a, block_b], 100, 1.0)

    system = System(mesh, cell, [polymer])
    system.set_interaction(0, 1, 0.25)

    return system


def test_cell_updater_creation():
    """Test that CellUpdater can be created with default parameters."""
    updater = CellUpdater(1.0)
    assert updater.damping == 1.0
    assert updater.stress_tolerance == 1e-6


def test_cell_updater_setters():
    """Test CellUpdater property setters."""
    updater = CellUpdater(1.0)
    updater.set_stress_tolerance(1e-8)
    assert updater.stress_tolerance == 1e-8


def test_cell_updater_step(diblock_system):
    """Test that CellUpdater.step() can be called."""
    system = diblock_system
    system.update()

    updater = CellUpdater(0.1)

    # Should be able to call step without error
    updater.step(system)


def test_stress_caching(diblock_system):
    """Test that stress is cached after system.update()."""
    system = diblock_system

    # Before update, stress might not be cached (empty list)
    system.update()

    # After update, stress should be available
    stress = system.stress()
    assert len(stress) == 1  # 1D lamellar has 1 component


def test_alternating_field_cell_updates(diblock_system):
    """Test alternating field and cell updates in a simple optimization loop."""
    system = diblock_system
    system.sample_fields(scale=0.1, seed=42)
    system.update()

    field_updater = FieldUpdater(system, step_size=0.01)
    cell_updater = CellUpdater(damping=0.5)

    # Run a few iterations with alternating updates
    for i in range(5):
        # 3 field steps
        for _ in range(3):
            field_updater.step(system)

        # 1 cell step
        cell_updater.step(system)

        # Check that stress is available
        stress = system.stress()
        assert len(stress) == 1

    # After a few iterations, system should still be valid
    final_stress = system.stress()
    assert len(final_stress) == 1


def test_field_error_decreases(diblock_system):
    """Test that field error decreases during field optimization."""
    system = diblock_system
    system.sample_fields(scale=0.1, seed=42)
    system.update()

    field_updater = FieldUpdater(system, step_size=0.01)

    initial_error = system.field_error()

    # Run field updates
    for _ in range(10):
        field_updater.step(system)

    final_error = system.field_error()

    # Field error should decrease
    assert final_error < initial_error


def test_cell_updater_convergence():
    """Test that CellUpdater can detect convergence when stress is low."""
    mesh = Mesh.One(32)
    cell = LamellarCell(10.0)

    # Create simple homogeneous system with low stress
    monomer = Monomer(0, 1.0)
    block = Block(monomer, 100, 1.0)
    polymer = Polymer([block], 100, 1.0)

    system = System(mesh, cell, [polymer])
    system.update()

    # Set very high tolerance so it converges immediately
    updater = CellUpdater(1.0)
    updater.set_stress_tolerance(10.0)

    # Should already be converged with high tolerance
    assert updater.converged(system) is True


def test_variable_cell_scft_simple():
    """Test a simple variable cell SCFT optimization loop."""
    mesh = Mesh.One(32)
    cell = LamellarCell(8.0)

    monomer_a = Monomer(0, 1.0)
    monomer_b = Monomer(1, 1.0)

    block_a = Block(monomer_a, 50, 1.0)
    block_b = Block(monomer_b, 50, 1.0)

    polymer = Polymer([block_a, block_b], 100, 1.0)

    system = System(mesh, cell, [polymer])
    system.set_interaction(0, 1, 0.25)
    system.sample_fields(scale=0.1, seed=123)
    system.update()

    field_updater = FieldUpdater(system, step_size=0.01)
    cell_updater = CellUpdater(damping=0.5)

    # Configurable alternation
    field_steps_per_cell_step = 5

    # Run optimization loop
    for i in range(20):
        # Update fields
        for _ in range(field_steps_per_cell_step):
            field_updater.step(system)

        # Update cell
        cell_updater.step(system)

        # Early stopping if cell converges
        if cell_updater.converged(system):
            break

    # Check that system is still valid
    stress = system.stress()
    assert len(stress) == 1
    error = system.field_error()
    assert error >= 0.0
