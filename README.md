# FTS

Polymer field-theoretic simulations in Rust and Python.

The repository consists of two main components:
1) The core library written in Rust, located in the [src](src/) folder.
2) The [pyfts](pyfts) module that depends on [pyo3](https://pyo3.rs/v0.21.2/) bindings to the Rust core.

> [!NOTE]
> This is an experimental project in order to to play around with
> developing Python extension modules in Rust.
> The core library is missing plenty of features compared to existing implementations,
> but may still serve as a useful guide for future developers.

## Installation

Install the [rustup](https://rustup.rs/) toolchain manager
and then install the [Maturin](https://www.maturin.rs/) build tool
in a Python virutal environment of your choosing.

### Local development

Install an editable version of the Python package by running:

```
# Use profile=release for optimized compilation
maturin develop --profile <profile>
```

A Python wheel can be built by the command following:

```
# Use profile=release for optimized compilation
maturin build --profile <profile> --release --out <output-directory>
```

## Performance

In order to simplify the Python extension module,
the majority of methods exposed in the Python API return copies of the underlying Rust types.
This means that accessing a field and mutating it within Python 
will not update the state of the underlying Rust objects.

In most cases, creating copies of these objects to return to Python has a negligible performance cost.
One area where this can become an issue is accessing the field/concentration arrays within a system.
Sending that data from Rust to Python involves creating a copy of the underlying arrays,
and doing so repeatedly may incur a significant overhead.
This is why system/field updates in a simulation loop 
are performed by dispatching to the underlying Rust types,
which can mutate the internal state of the system without copying data.

An example is shown below:

```
system = System(...)
mesh = system.mesh # Creates a cheap copy of the Mesh object
system.fields() # Creates copies of the field arrays

updater = FieldUpdater(...)
updater.step(system) # Mutates the Rust internals in-place
```

## References

1) [Polymer Self-Consistent Field Theory (PSCF)](https://github.com/dmorse/pscfpp/tree/master)
2) [The Equilibrium Theory of Inhomogeneous Polymers](https://academic.oup.com/book/34783)
3) [Field-Theoretic Simulations in Soft Matter and Quantum Fluids](https://academic.oup.com/book/45705)
