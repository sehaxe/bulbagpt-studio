use pyo3::prelude::*;
use std::fs::File;
use std::io::{Read, Write, BufWriter};
use std::path::Path;
use tokenizers::{Tokenizer, PostProcessorWrapper};
use std::sync::{Arc, mpsc};
use std::thread;
use rayon::prelude::*;

// Размер блока чтения (64 MB).
// Больше = быстрее (меньше IO вызовов), но больше потребление RAM.
const CHUNK_SIZE: usize = 64 * 1024 * 1024; 

#[pyfunction]
pub fn process_parallel(
    py: Python,
    files: Vec<String>,
    tokenizer_path: String,
    output_path: String
) -> PyResult<String> {
    
    py.allow_threads(move || {
        println!("RUST: 🚀 STARTING ENGINE (Chunked Read + Background Writer)...");

        // 1. Настройка токенизатора
        let mut tokenizer_raw = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;
        tokenizer_raw.with_post_processor(None::<PostProcessorWrapper>);
        let tokenizer = Arc::new(tokenizer_raw);

        // 2. Создаем канал для передачи данных Writer-потоку
        // sync_channel с буфером, чтобы не забить всю память, если диск медленный
        let (tx, rx) = mpsc::sync_channel::<Vec<u8>>(16);

        // 3. Запускаем Writer в отдельном потоке
        let output_path_clone = output_path.clone();
        let writer_handle = thread::spawn(move || -> std::io::Result<usize> {
            let file = File::create(output_path_clone)?;
            // Огромный буфер записи (32 MB)
            let mut writer = BufWriter::with_capacity(32 * 1024 * 1024, file);
            let mut total_tokens = 0;

            for bytes in rx {
                writer.write_all(&bytes)?;
                // Каждое число u16 занимает 2 байта
                total_tokens += bytes.len() / 2;
            }
            writer.flush()?;
            Ok(total_tokens)
        });

        // 4. Основной цикл по файлам
        let mut buffer = vec![0u8; CHUNK_SIZE];

        for path_str in files {
            let path = Path::new(&path_str);
            let mut file = match File::open(path) { Ok(f) => f, Err(_) => continue };
            
            let mut leftovers = Vec::new();

            loop {
                // Читаем кусок файла в буфер
                let bytes_read = match file.read(&mut buffer) {
                    Ok(0) => break, // EOF
                    Ok(n) => n,
                    Err(_) => break,
                };

                let chunk = &buffer[..bytes_read];

                // Нам нужно найти последний перенос строки, чтобы не разрезать строку посередине
                let (valid_chunk, rest) = if bytes_read == CHUNK_SIZE {
                    // Ищем последний '\n'
                    match chunk.iter().rposition(|&b| b == b'\n') {
                        Some(pos) => (&chunk[..=pos], &chunk[pos+1..]),
                        None => (chunk, &[][..]), // Очень длинная строка или конец, обрабатываем как есть
                    }
                } else {
                    (chunk, &[][..]) // Конец файла
                };

                // Собираем текст: "остатки с прошлого раза" + "текущий валидный кусок"
                // unsafe используется для скорости (мы предполагаем, что файлы валидный UTF-8).
                // Если файлы могут быть битыми, используйте String::from_utf8_lossy
                let text_to_process = if !leftovers.is_empty() {
                    leftovers.extend_from_slice(valid_chunk);
                    unsafe { String::from_utf8_unchecked(leftovers.clone()) }
                } else {
                    unsafe { String::from_utf8_unchecked(valid_chunk.to_vec()) }
                };

                // Сохраняем "хвост" для следующей итерации
                leftovers = rest.to_vec();
                if bytes_read < CHUNK_SIZE && leftovers.is_empty() {
                    // Если это был конец файла и хвостов нет, обрабатываем остаток и выходим
                }

                if text_to_process.trim().is_empty() {
                    if bytes_read < CHUNK_SIZE { break; } // EOF
                    continue; 
                }

                // 🔥 PARALLEL PROCESSING 🔥
                // Rayon сам разобьет text_to_process на строки (par_lines) без лишних аллокаций
                let processed_chunk: Vec<u8> = text_to_process
                    .par_lines()
                    .flat_map(|line| {
                        if line.is_empty() { return Vec::new(); }
                        
                        // Токенизация
                        if let Ok(encoding) = tokenizer.encode(line, false) {
                            let ids = encoding.get_ids();
                            // Конвертация u32 -> u16 "на лету" и сразу в байты
                            let mut byte_buf = Vec::with_capacity(ids.len() * 2);
                            for &id in ids {
                                if id != 0 {
                                    let id_u16 = id as u16;
                                    byte_buf.extend_from_slice(&id_u16.to_ne_bytes()); // Native Endian
                                }
                            }
                            byte_buf
                        } else {
                            Vec::new()
                        }
                    })
                    .collect(); // Собираем один большой бинарный блоб

                // Отправляем блоб писателю
                if !processed_chunk.is_empty() {
                    if tx.send(processed_chunk).is_err() {
                        break; // Writer умер
                    }
                }

                if bytes_read < CHUNK_SIZE {
                    break; // Конец файла
                }
            }
        }

        // Закрываем канал, чтобы writer понял, что данных больше не будет
        drop(tx);

        // Ждем завершения записи
        let final_count = writer_handle.join().unwrap().unwrap_or(0);
        
        println!("RUST: ✅ DONE. Total tokens: {}", final_count);
        Ok(format!("{}", final_count))
    })
}