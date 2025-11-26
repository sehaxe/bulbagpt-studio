use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::fs::File;
use memmap2::{Mmap, MmapOptions};

#[pyclass]
pub struct RustLinearLoader {
    mmap: Mmap,
    file_size: usize,
    cursor: usize,
}

#[pymethods]
impl RustLinearLoader {
    #[new]
    fn new(filename: String) -> PyResult<Self> {
        println!("RUST: 🚀 Initializing Linear Loader (Zero RAM overhead)...");
        let file = File::open(&filename)?;
        let file_size = file.metadata()?.len() as usize;
        
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        
        // Падказка OS: мы будзем чытаць паслядоўна
        #[cfg(unix)]
        unsafe {
            libc::madvise(mmap.as_ptr() as *mut _, file_size, libc::MADV_SEQUENTIAL);
        }

        Ok(RustLinearLoader {
            mmap,
            file_size,
            cursor: 0,
        })
    }

    fn next_batch<'py>(&mut self, py: Python<'py>, batch_size: usize, context_len: usize) -> PyResult<Bound<'py, PyBytes>> {
        let needed_tokens = batch_size * (context_len + 1);
        let bytes_needed = needed_tokens * 2;

        // Зацыкліванне (Epoch loop)
        if self.cursor + bytes_needed > self.file_size {
            self.cursor = 0;
            println!("RUST: 🔄 Epoch finished, rewinding...");
        }

        let start = self.cursor;
        let end = start + bytes_needed;
        
        let slice = &self.mmap[start..end];
        self.cursor = end;

        Ok(PyBytes::new_bound(py, slice))
    }
}