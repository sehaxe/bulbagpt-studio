use pyo3::prelude::*;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use rand::Rng;
use serde::Serialize;
use tokenizers::models::bpe::{BpeTrainerBuilder, BPE};
use tokenizers::models::ModelWrapper;
use tokenizers::models::TrainerWrapper;
use tokenizers::normalizers::{NormalizerWrapper, Sequence, NFC, Replace};
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::decoders::byte_level::ByteLevel as ByteLevelDecoder;
use tokenizers::processors::template::TemplateProcessing;
use tokenizers::{
    AddedToken, Tokenizer, 
    DecoderWrapper, PreTokenizerWrapper, PostProcessorWrapper
};

// 100 MB Safe Limit
const TARGET_TRAIN_SIZE: u64 = 100 * 1024 * 1024; 
const TEMP_DIR: &str = "system/temp_convert";
const TEMP_FILE: &str = "tokenizer_train_safe.txt";

#[derive(Serialize)]
struct TokenizerConfig {
    bos_token: String,
    eos_token: String,
    pad_token: String,
    model_max_length: usize,
    tokenizer_class: String,
    clean_up_tokenization_spaces: bool,
    add_bos_token: bool,
    add_eos_token: bool,
    use_default_system_prompt: bool,
}

#[derive(Serialize)]
struct SpecialTokensMap {
    bos_token: String,
    eos_token: String,
    pad_token: String,
}

#[pyfunction]
pub fn train_tokenizer_rust(
    files: Vec<String>,
    vocab_size: u32,
    output_dir: String
) -> PyResult<()> {
    
    // 🔥 FIX: Загортваем у unsafe, бо гэта патрабаванне новага Rust
    unsafe {
        std::env::set_var("RAYON_NUM_THREADS", "4");
    }
    
    println!("RUST: 🦀 ALL-IN-ONE Training (Tokenizer + Configs).");

    if let Err(e) = fs::create_dir_all(&output_dir) {
        return Err(PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Cannot create dir: {}", e)));
    }

    let (files_to_train, is_temp) = prepare_safe_dataset(&files)?;
    println!("RUST: 🧠 Training on safe sample...");

    // --- SETUP ---
    let model = ModelWrapper::BPE(BPE::default());
    let mut tokenizer = Tokenizer::new(model);

    let normalizer = NormalizerWrapper::Sequence(Sequence::new(vec![
        NormalizerWrapper::NFC(NFC), 
        NormalizerWrapper::Replace(Replace::new(r"\s+", " ".to_string()).unwrap()),
    ]));
    tokenizer.with_normalizer(Some(normalizer));

    let pre_tokenizer = PreTokenizerWrapper::ByteLevel(ByteLevel::new(false, true, false));
    tokenizer.with_pre_tokenizer(Some(pre_tokenizer));
    
    let decoder = DecoderWrapper::ByteLevel(ByteLevelDecoder::default());
    tokenizer.with_decoder(Some(decoder));

    let special_tokens_list = vec![
        AddedToken::from("<|begin_of_text|>", true),
        AddedToken::from("<|end_of_text|>", true),
        AddedToken::from("<|start_header_id|>", true),
        AddedToken::from("<|end_header_id|>", true),
        AddedToken::from("<|eot_id|>", true),
        AddedToken::from("<|finetune_right_pad_id|>", true),
        AddedToken::from("<|python_tag|>", true),
    ];

    let trainer = BpeTrainerBuilder::new()
        .show_progress(true)
        .vocab_size(vocab_size as usize)
        .min_frequency(2)
        .special_tokens(special_tokens_list.clone())
        .max_token_length(Some(32))
        .build();

    let mut trainer_wrapper = TrainerWrapper::BpeTrainer(trainer);

    tokenizer.train_from_files(&mut trainer_wrapper, files_to_train.clone())
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Training failed: {}", e)))?;

    if is_temp { let _ = fs::remove_file(&files_to_train[0]); }

    // --- POST PROCESSOR ---
    let post_processor = TemplateProcessing::builder()
        .try_single("$A") 
        .unwrap()
        .try_pair("$A:0 $B:1")
        .unwrap()
        .special_tokens(vec![
            ("<|begin_of_text|>", tokenizer.token_to_id("<|begin_of_text|>").unwrap()),
            ("<|end_of_text|>", tokenizer.token_to_id("<|end_of_text|>").unwrap()),
        ])
        .build()
        .unwrap();
    tokenizer.with_post_processor(Some(PostProcessorWrapper::Template(post_processor)));

    // --- SAVE ---
    let json_path = Path::new(&output_dir).join("tokenizer.json");
    tokenizer.save(json_path.to_str().unwrap(), true)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Save json failed: {}", e)))?;

    let config = TokenizerConfig {
        bos_token: "<|begin_of_text|>".to_string(),
        eos_token: "<|end_of_text|>".to_string(),
        pad_token: "<|finetune_right_pad_id|>".to_string(),
        model_max_length: 2048,
        tokenizer_class: "LlamaTokenizerFast".to_string(),
        clean_up_tokenization_spaces: false,
        add_bos_token: false, 
        add_eos_token: false,
        use_default_system_prompt: false,
    };
    let config_path = Path::new(&output_dir).join("tokenizer_config.json");
    let config_file = File::create(config_path)?;
    serde_json::to_writer_pretty(config_file, &config)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Save config failed: {}", e)))?;

    let map = SpecialTokensMap {
        bos_token: "<|begin_of_text|>".to_string(),
        eos_token: "<|end_of_text|>".to_string(),
        pad_token: "<|finetune_right_pad_id|>".to_string(),
    };
    let map_path = Path::new(&output_dir).join("special_tokens_map.json");
    let map_file = File::create(map_path)?;
    serde_json::to_writer_pretty(map_file, &map)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Save map failed: {}", e)))?;

    println!("RUST: ✅ All tokenizer files saved to '{}'", output_dir);
    Ok(())
}

fn prepare_safe_dataset(files: &Vec<String>) -> PyResult<(Vec<String>, bool)> {
    let mut total_size: u64 = 0;
    for f in files {
        if let Ok(meta) = fs::metadata(f) {
            total_size += meta.len();
        }
    }

    if total_size < TARGET_TRAIN_SIZE {
        return Ok((files.clone(), false));
    }

    if let Err(e) = fs::create_dir_all(TEMP_DIR) {
         return Err(PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Could not create temp dir: {}", e)));
    }

    let temp_path_buf = Path::new(TEMP_DIR).join(TEMP_FILE);
    let temp_path_str = temp_path_buf.to_string_lossy().to_string();
    let out_file = File::create(&temp_path_buf)?;
    let mut writer = BufWriter::new(out_file);
    
    let keep_ratio = TARGET_TRAIN_SIZE as f64 / total_size as f64;
    // FIX: Тут таксама выкарыстоўваем rng()
    let mut rng = rand::rng(); 
    let mut written_bytes = 0;

    for path_str in files {
        let file = match File::open(path_str) { Ok(f) => f, Err(_) => continue };
        let reader = BufReader::new(file);
        for line in reader.lines() {
            if let Ok(l) = line {
                if l.trim().is_empty() { continue; }
                if rng.random::<f64>() < keep_ratio {
                    writeln!(writer, "{}", l)?;
                    written_bytes += l.len() as u64;
                    if written_bytes >= (TARGET_TRAIN_SIZE + 5 * 1024 * 1024) { break; }
                }
            }
        }
    }
    writer.flush()?;
    Ok((vec![temp_path_str], true))
}