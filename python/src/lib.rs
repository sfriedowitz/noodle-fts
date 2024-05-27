use pyo3::prelude::*;

#[pyfunction]
fn hello(name: String) {
    println!("Hello from Rust, {name}")
}

#[pymodule]
#[pyo3(name = "_pyfts")]
fn extension_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello, m)?)?;
    Ok(())
}
