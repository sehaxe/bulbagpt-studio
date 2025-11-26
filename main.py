import os
import sys
import gc
import time
import math
import glob
import shutil
import datetime
import threading
import subprocess
import requests
import json
import re  # <--- Для парсинга номера шага
import pandas as pd
import numpy as np
import torch
import gradio as gr
import sentencepiece as spm

# 🔥 OPTIMIZATIONS: High Precision for Ampere+ GPUs
torch.set_float32_matmul_precision('high')

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["OMP_NUM_THREADS"] = "8"

from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizerFast
from safetensors.torch import save_file, load_file # <--- load_file для загрузки

# --- CHECKS ---
try:
    import bulba_rust
    HAS_RUST = True
except ImportError:
    print("❌ CRITICAL: 'bulba_rust' not found! Run: maturin develop --release")
    HAS_RUST = False

try:
    import bitsandbytes as bnb
    HAS_BNB = True
except ImportError: HAS_BNB = False

# --- DIRS ---
DIRS = {
    "DATA": "data",
    "CHECKPOINTS": "checkpoints",
    "MODELS": "output_models",
    "SYSTEM": "system",
    "TOKENIZER": os.path.join("system", "tokenizer"),
    "TOOLS": os.path.join("system", "tools"),
    "TEMP": os.path.join("system", "temp_convert"),
    "TEMP_TRAIN": os.path.join("system", "temp_train_tok")
}
for d in DIRS.values(): os.makedirs(d, exist_ok=True)
TRAIN_BIN_PATH = os.path.join(DIRS["DATA"], "train.bin")

# 🔥 PRESETS
PRESETS = {
    "Krolik (50M)":  { "h": 512,  "i": 1376, "l": 8,  "hd": 8,  "kv": 4, "ctx": 512, "bs": 8, "acc": 8, "steps": 5000 },
    "AIst (150M)":   { "h": 768,  "i": 2048, "l": 12, "hd": 12, "kv": 4, "ctx": 512, "bs": 8, "acc": 8, "steps": 10000 },
    "Zubr (350M)":   { "h": 1024, "i": 2816, "l": 28, "hd": 16, "kv": 4, "ctx": 512, "bs": 2, "acc": 32, "steps": 15000 },
}

def format_size(size_bytes):
    if size_bytes == 0: return "0 B"
    i = int(math.floor(math.log(size_bytes, 1024)))
    return "%s %s" % (round(size_bytes / math.pow(1024, i), 2), ("B", "KB", "MB", "GB")[i])

def get_lr(step, max_steps, max_lr):
    warmup = int(max_steps * 0.05)
    if step < warmup: return max_lr * (step + 1) / warmup
    if step > max_steps: return max_lr * 0.1
    decay = (step - warmup) / (max_steps - warmup)
    return max_lr * 0.1 + 0.5 * (1.0 + math.cos(math.pi * decay)) * (max_lr - max_lr * 0.1)

def check_system_status():
    bin_exists = os.path.exists(TRAIN_BIN_PATH)
    tok_path = os.path.join(DIRS["TOKENIZER"], "tokenizer.model")
    status = []
    if bin_exists: status.append(f"✅ Data: {format_size(os.path.getsize(TRAIN_BIN_PATH))}")
    else: status.append("❌ Data Missing")
    if os.path.exists(tok_path): status.append("✅ SentencePiece Ready")
    else: status.append("❌ Tokenizer Missing")
    return " | ".join(status)

def scan_checkpoints():
    files = sorted(glob.glob(f"{DIRS['CHECKPOINTS']}/*.safetensors"), key=os.path.getmtime, reverse=True)
    if not files: return gr.Dropdown(choices=[], value=None, label="No checkpoints found")
    return gr.Dropdown(choices=files, value=None, label="Resume from Checkpoint (Optional)")

# ================= 🆕 SENTENCEPIECE TRAINING =================

def train_sentencepiece_model(files, vocab_size, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(DIRS["TEMP_TRAIN"], exist_ok=True)
    
    print("📝 Preparing data for tokenizer training...")
    big_text_file = os.path.join(DIRS["TEMP_TRAIN"], "corpus.txt")
    total_written = 0
    limit = 100 * 1024 * 1024 
    
    with open(big_text_file, "w", encoding="utf-8") as out:
        for fpath in files:
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        if line.strip():
                            out.write(line)
                            total_written += len(line.encode("utf-8"))
                            if total_written > limit: break
            except: pass
            if total_written > limit: break
            
    print("🧠 Training SentencePiece BPE...")
    model_prefix = os.path.join(output_dir, "tokenizer")
    
    spm.SentencePieceTrainer.train(
        input=big_text_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        user_defined_symbols=["<|pad|>", "<|bos|>", "<|eos|>"],
        byte_fallback=True,
        train_extremely_large_corpus=False,
        character_coverage=1.0,
        unk_id=0, bos_id=1, eos_id=2, pad_id=-1
    )
    
    print("🔄 Converting to HuggingFace Fast format...")
    tokenizer = LlamaTokenizerFast(vocab_file=model_prefix + ".model")
    tokenizer.bos_token = "<s>"
    tokenizer.eos_token = "</s>"
    tokenizer.unk_token = "<unk>"
    tokenizer.pad_token = "<|pad|>"
    
    tokenizer.save_pretrained(output_dir)
    shutil.rmtree(DIRS["TEMP_TRAIN"], ignore_errors=True)
    print(f"✅ Tokenizer saved to {output_dir}")

# ================= 🚀 PROCESSING =================

def process_data(upload_files, local_path, progress=gr.Progress()):
    """
    1. Тренирует токенизатор (если нет).
    2. Конвертирует текст в токены (через Rust).
    3. ПЕРЕМЕШИВАЕТ данные (через Rust).
    """
    if not HAS_RUST: 
        return "❌ Rust Missing", None, 0, pd.DataFrame(), "Error"
    
    gc.collect()
    
    # --- 1. Сбор файлов ---
    files = []
    search_path = local_path if local_path else DIRS["DATA"]
    if upload_files: 
        files = [f.name for f in upload_files]
    elif os.path.exists(search_path):
        if os.path.isfile(search_path): 
            files = [search_path]
        else:
            for ext in ['.txt', '.json', '.md', '.csv', '.py']:
                files.extend(glob.glob(os.path.join(search_path, "**", ext), recursive=True))

    if not files: 
        return f"⚠️ No files in {search_path}", TRAIN_BIN_PATH, 0, pd.DataFrame(), "Error"

    # --- 2. Проверка токенизатора ---
    tok_dir = DIRS["TOKENIZER"]
    if not os.path.exists(os.path.join(tok_dir, "tokenizer.model")):
        progress(0.1, desc="🇧🇾 Training SentencePiece...")
        try:
            train_sentencepiece_model(files, 32000, tok_dir)
        except Exception as e:
            return f"❌ SPM Train Error: {e}", None, 0, pd.DataFrame(), "Error"

    # --- 3. Токенизация (Rust) ---
    progress(0.3, desc="🇧🇾 Rust: Tokenizing...")
    
    # Удаляем старый файл обучения перед началом
    if os.path.exists(TRAIN_BIN_PATH): 
        os.remove(TRAIN_BIN_PATH)

    abs_files = [os.path.abspath(f) for f in files]
    abs_tok = os.path.abspath(os.path.join(tok_dir, "tokenizer.json"))
    abs_out = os.path.abspath(TRAIN_BIN_PATH)

    task = {"done": False, "tokens": 0}
    
    def run_rust():
        try:
            # Запускаем Rust токенизатор
            res = bulba_rust.process_parallel(abs_files, abs_tok, abs_out)
            task["tokens"] = int(res)
        except Exception as e: 
            print(f"Rust Error: {e}")
        finally: 
            task["done"] = True

    t = threading.Thread(target=run_rust)
    t.start()
    
    hist_data = []
    start_time = time.time()
    
    # Ожидание завершения токенизации с обновлением графика
    while t.is_alive():
        time.sleep(0.5)
        elapsed = time.time() - start_time
        if elapsed < 0.1: continue
        
        if os.path.exists(TRAIN_BIN_PATH):
            cur_sz = os.path.getsize(TRAIN_BIN_PATH)
            avg_speed = (cur_sz / (1024*1024)) / elapsed
            hist_data.append({"time": elapsed, "speed": avg_speed})
            
            if len(hist_data) > 100: display_data = hist_data[-100:]
            else: display_data = hist_data
                
            df = pd.DataFrame(display_data)
            yield f"Tokenizing... {format_size(cur_sz)} | Avg: {avg_speed:.1f} MB/s", TRAIN_BIN_PATH, 0, df, "Working..."
        
    t.join() # Поток завершен

    # --- 4. ПЕРЕМЕШИВАНИЕ (SHUFFLE) ---
    # Это ключевой момент для исправления скачков Loss!
    if task["done"] and os.path.exists(TRAIN_BIN_PATH):
        try:
            progress(0.9, desc="🎲 Shuffling dataset (Rust)...")
            yield "🎲 Shuffling dataset... Please wait...", TRAIN_BIN_PATH, 0, pd.DataFrame(hist_data), "Shuffling..."
            
            # Определяем EOS ID (Обычно 2 для SentencePiece)
            eos_id = 2
            try:
                # Пытаемся считать реальный EOS ID из tokenizer.json
                with open(abs_tok, 'r', encoding='utf-8') as f:
                    tj = json.load(f)
                    # Структура может отличаться, но часто это model -> eos_token_id
                    # Если не найдем - оставим 2
                    pass 
            except: pass
            
            # Переименовываем train.bin -> temp_unshuffled.bin
            temp_bin = os.path.join(DIRS["DATA"], "temp_unshuffled.bin")
            if os.path.exists(temp_bin): os.remove(temp_bin)
            os.rename(TRAIN_BIN_PATH, temp_bin)
            
            print(f"🎲 Running Rust Shuffle on {format_size(os.path.getsize(temp_bin))}...")
            
            # Вызываем функцию из shuffler.rs
            bulba_rust.shuffle_dataset(temp_bin, TRAIN_BIN_PATH, eos_id)
            
            # Удаляем временный файл
            if os.path.exists(temp_bin): os.remove(temp_bin)
            print("✅ Shuffle done!")
            
        except Exception as e:
            print(f"❌ Shuffle Warning: {e}")
            # Если перемешивание упало, вернем файл назад
            if os.path.exists(temp_bin) and not os.path.exists(TRAIN_BIN_PATH):
                os.rename(temp_bin, TRAIN_BIN_PATH)
    
    final_sz = os.path.getsize(TRAIN_BIN_PATH) if os.path.exists(TRAIN_BIN_PATH) else 0
    yield f"✅ DONE! Tokens: {task['tokens']:,} | Size: {format_size(final_sz)}", TRAIN_BIN_PATH, 0, pd.DataFrame(hist_data), "Done"

# ================= 🧠 TRAINING LOOP =================

STOP_FLAG = False
SAVE_FLAG = False 

def save_safe(model, path):
    folder = os.path.dirname(path)
    if not os.path.exists(folder): os.makedirs(folder, exist_ok=True)
    raw = model._orig_mod if hasattr(model, "_orig_mod") else model
    sd = raw.state_dict()
    if "lm_head.weight" in sd and "model.embed_tokens.weight" in sd:
        if sd["lm_head.weight"].data_ptr() == sd["model.embed_tokens.weight"].data_ptr():
            del sd["lm_head.weight"]
    save_file(sd, path)
    print(f"💾 Saved: {path}")

def train_loop(train_path, model_name, lr, steps_ui, mode, resume_path, progress=gr.Progress()):
    global STOP_FLAG, SAVE_FLAG
    STOP_FLAG = False
    SAVE_FLAG = False
    
    gc.collect()
    torch.cuda.empty_cache()
    
    target_bin = train_path if train_path else TRAIN_BIN_PATH
    if not os.path.exists(target_bin): return f"❌ Data not found", None, pd.DataFrame()
    
    tok_path = DIRS["TOKENIZER"]
    try: 
        tok = LlamaTokenizerFast.from_pretrained(tok_path)
    except: return "❌ Tokenizer Error", None, pd.DataFrame()

    real_vocab_size = len(tok)
    print(f"📊 Vocab: {real_vocab_size}")

    p = PRESETS[model_name]
    max_steps = int(steps_ui)
    device = torch.device('cuda')
    
    conf = LlamaConfig(
        vocab_size=real_vocab_size, hidden_size=p["h"], intermediate_size=p["i"],
        num_hidden_layers=p["l"], num_attention_heads=p["hd"], num_key_value_heads=p["kv"],
        max_position_embeddings=p["ctx"], architectures=["LlamaForCausalLM"],
        attn_implementation="sdpa"
    )
    
    model = LlamaForCausalLM(conf).to(device)
    
    # === RESUME LOGIC ===
    start_step = 0
    if resume_path and os.path.exists(resume_path):
        print(f"🔄 Resuming from: {resume_path}")
        try:
            # Загружаем веса
            state_dict = load_file(resume_path)
            model.load_state_dict(state_dict, strict=False)
            
            # Пытаемся найти номер шага в названии файла (например step_500.safetensors)
            match = re.search(r"(\d+)", os.path.basename(resume_path))
            if match:
                start_step = int(match.group(1))
                print(f"⏩ Starting from step {start_step}")
            else:
                print("⚠️ Step number not found in filename, starting from 0 but with loaded weights.")
        except Exception as e:
            print(f"❌ Failed to load checkpoint: {e}")
            return f"❌ Resume Error: {e}", None, pd.DataFrame()
    else:
        print("🆕 Starting training from scratch")
    # ====================
    
    print("🚀 SPEED MODE: Gradient Checkpointing DISABLED (Fastest)")

    if "Compile" in mode or "Both" in mode:
        print("⚡ Compiling Model (reduce-overhead)...")
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except Exception as e:
            print(f"⚠️ Compile failed, continuing standard: {e}")

    model.train()

    if HAS_BNB:
        print("⚖️ Using 8-bit AdamW (Balance Speed/Mem)")
        optim = bnb.optim.AdamW8bit(model.parameters(), lr=lr)
    else:
        print("⚠️ BNB Missing. Using Fused AdamW")
        optim = torch.optim.AdamW(model.parameters(), lr=lr, fused=True)
    
    scaler = torch.amp.GradScaler()
    
    try: loader = bulba_rust.RustDataLoader(target_bin, tok.eos_token_id, True)
    except Exception as e: return f"❌ Loader Error: {e}", None, pd.DataFrame()

    hist_tr = []
    avg_loss = 0
    t0 = time.time()
    
    expected_len = p["bs"] * (p["ctx"] + 1)

    print(f"🚀 STARTING LOOP ({start_step} -> {max_steps})...")

    # 🔥 Loop starts from start_step
    for step in range(start_step, max_steps):
        if STOP_FLAG: break
        
        current_lr = get_lr(step, max_steps, lr)
        for g in optim.param_groups: g['lr'] = current_lr
        
        optim.zero_grad(set_to_none=True) 
        
        loss_accum = 0
        for _ in range(p["acc"]):
            raw = loader.next_batch(p["bs"], p["ctx"] + 1)
            
            data_np = np.frombuffer(raw, dtype=np.uint16).astype(np.int64)

            if data_np.size > expected_len:
                data_np = data_np[:expected_len]

            data = torch.from_numpy(data_np).view(p["bs"], p["ctx"] + 1).to(device, non_blocking=True)
            data.clamp_(max=real_vocab_size-1)

            with torch.amp.autocast("cuda"):
                loss = model(data[:, :-1], labels=data[:, 1:]).loss / p["acc"]
            
            scaler.scale(loss).backward()
            loss_accum += loss.item()
        
        scaler.unscale_(optim)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        scaler.step(optim)
        scaler.update()
        
        if step == start_step: avg_loss = loss_accum
        else: avg_loss = 0.95 * avg_loss + 0.05 * loss_accum
        
        if SAVE_FLAG:
            save_safe(model, f"{DIRS['CHECKPOINTS']}/manual_{step}.safetensors")
            SAVE_FLAG = False

        if step % 10 == 0:
            dt = time.time() - t0
            if dt == 0: dt = 0.001
            # Корректный расчет скорости с учетом стартового шага
            steps_done = step - start_step + 1
            spd = int(steps_done * p["bs"] * p["acc"] * p["ctx"] / dt)
            eta = str(datetime.timedelta(seconds=int((max_steps - step) / (steps_done/dt))))
            hist_tr.append({"step": step, "loss": avg_loss})
            
            if step % 500 == 0 and step > 0: 
                save_safe(model, f"{DIRS['CHECKPOINTS']}/step_{step}.safetensors")
                
            yield f"Step: {step} | Loss: {avg_loss:.4f} | Speed: {spd} tok/s | ETA: {eta}", pd.DataFrame(hist_tr)

    save_safe(model, f"{DIRS['CHECKPOINTS']}/final.safetensors")
    yield "✅ Done!", pd.DataFrame(hist_tr)

def stop_train(): global STOP_FLAG; STOP_FLAG = True; return "🛑 Stopping..."
def trigger_save(): global SAVE_FLAG; SAVE_FLAG = True; return "💾 Saving..." 

# ================= 📦 EXPORT (MANUAL & ROBUST) =================

def auto_export_gguf(selected_ckpt, model_name_ui):
    if not selected_ckpt: return "❌ No checkpoint selected"
    yield f"Selected: {selected_ckpt}"
    
    script_path = os.path.join(DIRS["TOOLS"], "convert_hf_to_gguf.py")
    if not os.path.exists(script_path):
        yield "📥 Downloading converter script..."
        try:
            url = "https://raw.githubusercontent.com/ggerganov/llama.cpp/master/convert_hf_to_gguf.py"
            response = requests.get(url)
            if response.status_code == 200:
                with open(script_path, "wb") as f: f.write(response.content)
            else: return "❌ Failed to download convert script"
        except Exception as e: return f"❌ Network error: {e}"

    hf_path = DIRS["TEMP"]
    if os.path.exists(hf_path): shutil.rmtree(hf_path)
    os.makedirs(hf_path, exist_ok=True)
    
    try:
        yield "📂 Preparing files..."
        
        tok_model_src = os.path.join(DIRS["TOKENIZER"], "tokenizer.model")
        if not os.path.exists(tok_model_src):
            return "❌ Tokenizer.model not found! Train tokenizer first."
        shutil.copy(tok_model_src, os.path.join(hf_path, "tokenizer.model"))

        p = PRESETS.get(model_name_ui, PRESETS["AIst (150M)"])
        
        manual_config = {
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "vocab_size": 32000,
            "hidden_size": p["h"],
            "intermediate_size": p["i"],
            "num_hidden_layers": p["l"],
            "num_attention_heads": p["hd"],
            "num_key_value_heads": p["kv"],
            "max_position_embeddings": p["ctx"],
            "rope_theta": 10000.0,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "rms_norm_eps": 1e-5
        }
        
        with open(os.path.join(hf_path, "config.json"), "w") as f:
            json.dump(manual_config, f, indent=2)

        shutil.copy(selected_ckpt, os.path.join(hf_path, "model.safetensors"))
        
        yield "🚀 Running GGUF conversion..."
        ckpt_name = os.path.basename(selected_ckpt).replace(".safetensors", "")
        out_file = os.path.join(DIRS["MODELS"], f"bulba_{ckpt_name}.gguf")
        
        cmd = [sys.executable, script_path, hf_path, "--outfile", out_file, "--outtype", "f16"]
        
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            encoding='utf-8', 
            errors='replace'
        )
        
        if result.returncode == 0:
            yield f"✅ SUCCESS!\nSaved to: {out_file}\n\n=== LOGS ===\n{result.stderr}"
        else:
            yield f"❌ FAILED!\n\n=== STDERR ===\n{result.stderr}\n\n=== STDOUT ===\n{result.stdout}"

    except Exception as e: yield f"❌ Python Error: {e}"

# ================= UI =================

def update_steps(name): return gr.Number(value=PRESETS[name]["steps"])

with gr.Blocks(title="BulbaGPT Studio") as demo:
    gr.Markdown("# 🥔 BulbaGPT Studio: Resume Edition")
    s_path = gr.State(TRAIN_BIN_PATH)
    s_data_dir = gr.State(DIRS["DATA"])

    with gr.Tabs():
        with gr.Tab("🚀 Train"):
            with gr.Row():
                with gr.Column():
                    files = gr.File(label="Files", file_count="multiple")
                    btn_proc = gr.Button("1. Prepare Data & Tokenizer", variant="secondary")
                    status_proc = gr.Textbox(label="Status")
                    plot_proc = gr.LinePlot(x="time", y="speed", title="Processing Speed (MB/s)", height=250)
                    
                    model_sel = gr.Radio(list(PRESETS.keys()), value="AIst (150M)", label="Config")
                    steps = gr.Number(label="Steps", value=10000)
                    model_sel.change(update_steps, model_sel, steps)
                    mode = gr.Dropdown(["Compile", "Flash Attention", "Both"], value="Both", label="Mode")
                    
                    # 🔥 NEW: Resume Option
                    with gr.Row():
                        resume_ckpt = gr.Dropdown(label="Resume from Checkpoint (Optional)", choices=[], value=None, scale=3)
                        btn_refresh_res = gr.Button("🔄", scale=1)

                    with gr.Row():
                        btn_run = gr.Button("2. START", variant="primary")
                        btn_save = gr.Button("Save")
                        btn_stop = gr.Button("Stop")
                with gr.Column():
                    status_train = gr.Textbox(label="Log")
                    plot = gr.LinePlot(x="step", y="loss", title="Loss", height=300)
        
        with gr.Tab("📦 Export"):
            ckpt = gr.Dropdown(label="Checkpoint")
            btn_ref = gr.Button("Refresh")
            btn_exp = gr.Button("Convert GGUF", variant="primary")
            log_exp = gr.Textbox()

    demo.load(check_system_status, outputs=status_proc)
    demo.load(scan_checkpoints, outputs=ckpt)
    demo.load(scan_checkpoints, outputs=resume_ckpt) # Scan checkpoints on load for resume
    
    btn_proc.click(process_data, [files, s_data_dir], [status_proc, s_path, gr.State(), plot_proc, gr.State()])
    
    # 🔥 Pass resume_ckpt to train_loop
    btn_run.click(train_loop, [s_path, model_sel, gr.Number(3e-4, visible=False), steps, mode, resume_ckpt], [status_train, plot])
    
    btn_stop.click(stop_train, outputs=status_train)
    btn_save.click(trigger_save, outputs=status_train) 
    
    btn_ref.click(scan_checkpoints, outputs=ckpt)
    btn_refresh_res.click(scan_checkpoints, outputs=resume_ckpt) # Button to refresh resume list
    
    btn_exp.click(auto_export_gguf, inputs=[ckpt, model_sel], outputs=log_exp)

if __name__ == "__main__":
    demo.queue().launch()