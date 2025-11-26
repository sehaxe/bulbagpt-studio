// src/shuffler.rs
use pyo3::prelude::*;
use std::fs::File;
use std::io::{Write, BufWriter};
use memmap2::MmapOptions;
use rand::seq::SliceRandom;
use rand::thread_rng; // Используем thread_rng для стандартной версии rand
use rayon::prelude::*;

#[pyfunction]
pub fn shuffle_dataset(input_path: String, output_path: String, eos_token_id: u16) -> PyResult<String> {
    println!("RUST: 🚀 Starting Hyper-Speed Shuffle...");
    println!("RUST: 📂 Input: {}", input_path);

    // 1. Открываем файл через Memory Map (это не грузит RAM, а создает "окно" в файл)
    let file = File::open(&input_path)?;
    let mmap = unsafe { MmapOptions::new().map(&file)? };

    // Проверка на пустой файл
    if mmap.len() == 0 {
        return Ok("Empty file".to_string());
    }

    // 2. Представляем байты как массив u16 (Zero-Copy)
    // ВАЖНО: Работает на Little Endian системах (x86, ARM)
    let u16_len = mmap.len() / 2;
    let tokens: &[u16] = unsafe {
        std::slice::from_raw_parts(mmap.as_ptr() as *const u16, u16_len)
    };

    println!("RUST: 🔍 Scanning {} tokens for EOS markers ({}) using Rayon...", u16_len, eos_token_id);

    // 3. Параллельно ищем индексы всех EOS токенов
    // Rayon использует все ядра процессора для поиска
    let mut eos_indices: Vec<usize> = tokens
        .par_iter()
        .enumerate()
        .filter_map(|(i, &token)| {
            if token == eos_token_id { Some(i) } else { None }
        })
        .collect();

    // Если файл не заканчивается EOS, добавляем конец файла как границу
    if eos_indices.last() != Some(&(u16_len - 1)) {
        eos_indices.push(u16_len - 1);
    }

    let total_sequences = eos_indices.len();
    println!("RUST: ✅ Found {} sequences. Preparing shuffle...", total_sequences);

    if total_sequences < 2 {
        println!("RUST: ⚠️ Not enough sequences to shuffle. Copying as is.");
        let mut out = File::create(&output_path)?;
        out.write_all(&mmap)?;
        return Ok("Skipped (too small)".to_string());
    }

    // 4. Собираем список диапазонов (Start Byte, Length Byte)
    // Используем байты (usize * 2), чтобы потом быстро писать
    let mut samples: Vec<(usize, usize)> = Vec::with_capacity(total_sequences);
    let mut start_idx = 0;

    for &end_idx in &eos_indices {
        let len_tokens = end_idx - start_idx + 1;
        // Конвертируем индексы токенов в индексы байтов (* 2)
        samples.push((start_idx * 2, len_tokens * 2));
        start_idx = end_idx + 1;
    }

    // 5. Перемешиваем список диапазонов
    println!("RUST: 🎲 Shuffling ranges...");
    let mut rng = thread_rng();
    samples.shuffle(&mut rng);

    // 6. Записываем новый файл
    // Буферизация 64MB значительно ускоряет запись на SSD/HDD
    println!("RUST: 💾 Writing to output: {}", output_path);
    let out_file = File::create(&output_path)?;
    let mut writer = BufWriter::with_capacity(64 * 1024 * 1024, out_file);

    // Мы просто копируем куски памяти в новый порядок
    for (start, len) in samples {
        // Проверка выхода за границы (на всякий случай)
        if start + len <= mmap.len() {
            writer.write_all(&mmap[start..start + len])?;
        }
    }

    writer.flush()?;
    println!("RUST: 🎉 Shuffle Complete!");
    Ok("Success".to_string())
}