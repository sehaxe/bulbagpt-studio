use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::fs::File;
use memmap2::{Mmap, MmapOptions};
use rand::seq::SliceRandom;
// FIX: use rand::rng()
use rand::rng; 

#[pyclass]
pub struct RustDataLoader {
    mmap: Mmap,
    samples: Vec<(u32, u32)>, 
    order: Vec<usize>,
    order_idx: usize,
    current_sample_offset: usize,
    shuffle: bool,
    cluster_size_bytes: usize,
}

#[pymethods]
impl RustDataLoader {
    #[new]
    fn new(filename: String, eos_token_id: u16, shuffle: bool) -> PyResult<Self> {
        println!("RUST: 🧹 Preparing Smart Loader...");
        let file = File::open(&filename)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };

        // 1. Скан
        let mut samples = Vec::new(); 
        let mut start = 0;
        
        for (i, chunk) in mmap.chunks_exact(2).enumerate() {
            let token = u16::from_le_bytes([chunk[0], chunk[1]]);
            if token == eos_token_id {
                let current_pos = (i + 1) * 2;
                let len = current_pos - start;
                if len > 0 {
                    samples.push((start as u32, len as u32));
                }
                start = current_pos; 
            }
        }
        if start < mmap.len() {
            samples.push((start as u32, (mmap.len() - start) as u32));
        }
        
        samples.shrink_to_fit();
        
        // 2. Кластарнае перамешванне
        let cluster_size_bytes = 64 * 1024 * 1024; 
        let mut order = Self::create_clustered_order(&samples, cluster_size_bytes, shuffle);
        order.shrink_to_fit();

        Ok(RustDataLoader {
            mmap,
            samples,
            order,
            order_idx: 0,
            current_sample_offset: 0,
            shuffle,
            cluster_size_bytes,
        })
    }

    fn next_batch<'py>(&mut self, py: Python<'py>, batch_size: usize, context_len: usize) -> PyResult<Bound<'py, PyBytes>> {
        let needed_tokens = batch_size * (context_len + 1);
        let mut needed_bytes = needed_tokens * 2;
        let mut buffer = Vec::with_capacity(needed_bytes);

        while needed_bytes > 0 {
            if self.order_idx >= self.order.len() {
                self.order_idx = 0;
                if self.shuffle {
                    self.order = Self::create_clustered_order(&self.samples, self.cluster_size_bytes, true);
                }
            }

            let sample_idx = self.order[self.order_idx];
            let (start_u32, len_u32) = self.samples[sample_idx];
            let start_byte = start_u32 as usize;
            let len_byte = len_u32 as usize;

            let remaining = len_byte - self.current_sample_offset;
            let to_take = std::cmp::min(remaining, needed_bytes);

            let read_start = start_byte + self.current_sample_offset;
            let read_end = read_start + to_take;
            
            buffer.extend_from_slice(&self.mmap[read_start..read_end]);

            #[cfg(unix)]
            unsafe {
                let page_size = 4096;
                let ptr = self.mmap.as_ptr();
                let addr_start = ptr as usize + read_start;
                let addr_end = ptr as usize + read_end;
                let pg_start = (addr_start + page_size - 1) & !(page_size - 1);
                let pg_end = addr_end & !(page_size - 1);
                if pg_end > pg_start {
                    libc::madvise(pg_start as *mut _, pg_end - pg_start, libc::MADV_DONTNEED);
                }
            }

            self.current_sample_offset += to_take;
            needed_bytes -= to_take;

            if self.current_sample_offset >= len_byte {
                self.order_idx += 1;
                self.current_sample_offset = 0;
            }
        }

        Ok(PyBytes::new_bound(py, &buffer))
    }
}

impl RustDataLoader {
    fn create_clustered_order(samples: &Vec<(u32, u32)>, cluster_limit: usize, shuffle: bool) -> Vec<usize> {
        let mut final_order = Vec::with_capacity(samples.len());
        if !shuffle {
            for i in 0..samples.len() { final_order.push(i); }
            return final_order;
        }

        let mut chunks: Vec<Vec<usize>> = Vec::new();
        let mut current_chunk = Vec::new();
        let mut current_size = 0;

        for (idx, &(_, len)) in samples.iter().enumerate() {
            current_chunk.push(idx);
            current_size += len as usize;
            if current_size >= cluster_limit {
                chunks.push(current_chunk);
                current_chunk = Vec::new();
                current_size = 0;
            }
        }
        if !current_chunk.is_empty() { chunks.push(current_chunk); }

        // FIX: Выкарыстоўваем rng()
        let mut rng = rng();
        chunks.shuffle(&mut rng);
        for mut chunk in chunks {
            chunk.shuffle(&mut rng);
            final_order.extend(chunk);
        }
        final_order
    }
}