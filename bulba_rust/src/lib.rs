use pyo3::prelude::*;

mod processor;
mod loader;
mod linear_loader;
mod shuffler;
mod trainer; // <--- ДАДАЛІ

use processor::process_parallel;
use loader::RustDataLoader;
use linear_loader::RustLinearLoader;
use shuffler::shuffle_dataset;
use trainer::train_tokenizer_rust; // <--- ДАДАЛІ

#[pymodule]
fn bulba_rust(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(process_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(shuffle_dataset, m)?)?;
    m.add_function(wrap_pyfunction!(train_tokenizer_rust, m)?)?; // <--- ДАДАЛІ
    m.add_class::<RustDataLoader>()?;
    m.add_class::<RustLinearLoader>()?;
    Ok(())
}