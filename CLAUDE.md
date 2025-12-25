# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Noodle FTS is a polymer field-theoretic simulation library with:
- Core library written in Rust (src/)
- Python bindings via PyO3 (pynoodle/)

The library implements self-consistent field theory (SCFT) for polymer systems with support for various unit cell geometries and molecular species types.

## Development Commands

### Building and Installation

```bash
# Install editable Python package (debug mode)
maturin develop

# Install with optimizations (recommended for testing)
maturin develop --profile release

# Build Python wheel
maturin build --profile release --out dist/
```

### Testing

```bash
# Run Rust tests
cargo test

# Run Python tests
pytest

# Run tests with coverage
pytest --cov=pynoodle
```

### Code Formatting

```bash
# Format Rust code (follows rustfmt.toml config)
cargo fmt

# Check formatting without modifying files
cargo fmt --check
```

### Linting

```bash
# Run Clippy for Rust linting
cargo clippy

# Run Clippy with all features
cargo clippy --all-features
```

## Architecture

### Core Module Structure

The Rust library is organized into several key modules:

- **chem**: Chemical components (Monomer, Block, Species)
  - Species is an enum over Point and Polymer types
  - Implements SpeciesDescription trait for computing system-level quantities

- **domain**: Spatial discretization (Mesh, UnitCell, Domain, FFT)
  - UnitCell is an enum supporting Lamellar, Square, Hexagonal2D, and Cubic geometries
  - Domain combines Mesh and UnitCell with FFT capabilities

- **solvers**: Field-theoretic solvers (SpeciesSolver, Propagator)
  - SpeciesSolver is an enum over PointSolver and PolymerSolver
  - Implements SolverOps trait for partition function calculation
  - PolymerSolver uses propagator methods for chain statistics

- **system**: System state and field updates (System, Interaction, FieldUpdater)
  - System manages monomers, species solvers, fields, concentrations, and potentials
  - Uses HashMap<usize, T> to store field data indexed by monomer IDs
  - FieldUpdater handles SCFT iteration steps

- **fields**: Type aliases and operations for multi-dimensional arrays
  - RField: Real-valued field (ArrayD<f64>)
  - CField: Complex-valued field (ArrayD<Complex64>)
  - FieldOps trait for common field operations

- **python**: PyO3 bindings exposing Rust types to Python
  - Enabled with "python" feature flag
  - Module registered as _core in pynoodle package

### Important Design Patterns

1. **enum_dispatch**: Used extensively for Species and SpeciesSolver to provide trait-based polymorphism with zero-cost abstraction

2. **HashMap storage**: Fields, concentrations, and potentials are stored in HashMaps indexed by arbitrary monomer IDs (not sequential arrays)

3. **Copy semantics for Python**: Python API returns copies of Rust objects for simplicity
   - Accessing System.fields() creates array copies
   - Mutating Python objects doesn't affect Rust internals
   - Use FieldUpdater.step(system) to mutate Rust state in-place

4. **Mesh-based arrays**: All field arrays have dynamic shapes matching the Mesh dimensions

### Python Extension Module

The Python package (pynoodle) exposes:
- Core types: Monomer, Block, Species (Point/Polymer)
- Domain types: Mesh, UnitCell variants (LamellarCell, SquareCell, etc.)
- System: Main simulation container
- FieldUpdater: SCFT iteration driver

Built with Maturin and uses numpy for array interop.

## Feature Flags

- `default`: Uses openblas-system
- `openblas`: Compile with OpenBLAS
- `openblas-system`: Use system-provided OpenBLAS
- `python`: Enable PyO3 bindings (required for pynoodle)

## Testing and Verification

**CRITICAL**: All new functionality must include test cases.

### Test Requirements

1. **Rust tests** (in `#[cfg(test)]` modules)
   - Unit tests for new functions/methods
   - Integration tests for cross-module functionality
   - Use `float_cmp::assert_approx_eq!` for floating-point comparisons
   - Test edge cases and error conditions

2. **Python tests** (in `tests/` directory)
   - Test Python API bindings
   - End-to-end workflow tests
   - Use pytest fixtures for common setup

3. **Verification approach**
   - Compare against analytical solutions when available
   - Test limiting cases (e.g., homogeneous limit, weak segregation)
   - Verify physical constraints (e.g., incompressibility, partition function = 1)
   - Check numerical stability and convergence

4. **Running tests**
   ```bash
   # Rust tests (run before committing)
   cargo test

   # Python tests (after maturin develop)
   pytest
   ```

## Code Style

Follow rustfmt.toml configuration:
- max_width = 110
- use_field_init_shorthand = true
- Imports: grouped by Std/External/Crate, reordered automatically

### Python Code Style

- **Never use** `from __future__ import annotations` - causes issues with type checking
- **Always use absolute imports**, never relative imports (e.g., `from pynoodle.configs import X` not `from .configs import X`)
- Use type hints directly without string quotes when possible
- Follow ruff formatting and linting rules in pyproject.toml
- **Never add docstrings to test functions or classes** - test names should be self-explanatory
