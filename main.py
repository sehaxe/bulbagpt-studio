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
import re
import atexit
import pandas as pd
import numpy as np
import torch
import gradio as gr
import sentencepiece as spm
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizerFast
from safetensors.torch import save_file, load_file

# ================= ⚙️ HARDWARE SETUP =================

IS_MAC = sys.platform == "darwin"
DEVICE = "cpu"

if torch.cuda.is_available():
    DEVICE = "cuda"
    torch.set_float32_matmul_precision('high')
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    print("🚀 Hardware: NVIDIA CUDA Detected")
elif torch.backends.mps.is_available():
    DEVICE = "mps"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    print("🍎 Hardware: Apple Silicon (Metal/MPS) Detected")
else:
    print("🐌 Hardware: CPU Only (Warning: Slow)")

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["OMP_NUM_THREADS"] = "8"

# --- CHECKS ---
HAS_RUST = False
try:
    import bulba_rust
    HAS_RUST = True
except ImportError:
    print("⚠️ WARNING: 'bulba_rust' module not found. Processing will be slower or fail.")

HAS_BNB = False
if DEVICE == "cuda":
    try:
        import bitsandbytes as bnb
        HAS_BNB = True
    except ImportError: HAS_BNB = False

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TB = True
except ImportError:
    HAS_TB = False

# --- DIRS ---
DIRS = {
    "DATA": "data",
    "CHECKPOINTS": "checkpoints",
    "MODELS": "output_models",
    "SYSTEM": "system",
    "TOKENIZER": os.path.join("system", "tokenizer"),
    "TOOLS": os.path.join("system", "tools"),
    "TEMP": os.path.join("system", "temp_convert"),
    "TEMP_TRAIN": os.path.join("system", "temp_train_tok"),
    "LOGS": "runs"
}
for d in DIRS.values(): os.makedirs(d, exist_ok=True)
TRAIN_BIN_PATH = os.path.join(DIRS["DATA"], "train.bin")

# 🔥 PRESETS
PRESETS = {
    "Krolik (45M)":  { "h": 512, "i": 1376, "l": 8, "hd": 8, "kv": 4, "ctx": 512, "bs": 16, "acc": 4, "steps": 5000, "lr": 8e-4 },
    "Vorona (110M)": { "h": 768, "i": 2048, "l": 12, "hd": 12, "kv": 4, "ctx": 512, "bs": 8, "acc": 8, "steps": 8000, "lr": 6e-4 },
    "AIst (250M)":   { "h": 1024, "i": 2816, "l": 16, "hd": 16, "kv": 4, "ctx": 512, "bs": 4, "acc": 16, "steps": 12000, "lr": 4e-4 },
    "Zubr (600M)":   { "h": 1536, "i": 4096, "l": 22, "hd": 24, "kv": 8, "ctx": 512, "bs": 1, "acc": 64, "steps": 18000, "lr": 3e-4 },
}

# ================= 📊 TENSORBOARD MANAGER =================

tb_process = None

def start_tensorboard_background():
    global tb_process
    if not HAS_TB or tb_process is not None: return
    print("📊 Starting TensorBoard...")
    log_dir = os.path.abspath(DIRS["LOGS"])
    cmd = [sys.executable, "-m", "tensorboard.main", "--logdir", log_dir, "--port", "6006", "--host", "0.0.0.0"]
    try:
        tb_process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"❌ Failed to start TensorBoard: {e}")

def kill_tensorboard():
    global tb_process
    if tb_process:
        tb_process.terminate()
        tb_process = None

atexit.register(kill_tensorboard)

# ================= HELPER FUNCTIONS =================

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
    limit = 50 * 1024 * 1024 # Limit 50MB for tokenizer training
    
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
    
    tokenizer = LlamaTokenizerFast(vocab_file=model_prefix + ".model")
    tokenizer.bos_token = "<s>"
    tokenizer.eos_token = "</s>"
    tokenizer.unk_token = "<unk>"
    tokenizer.pad_token = "<|pad|>"
    tokenizer.save_pretrained(output_dir)
    shutil.rmtree(DIRS["TEMP_TRAIN"], ignore_errors=True)

# ================= 🚀 PROCESSING =================

def process_data(upload_files, local_path, progress=gr.Progress()):
    # ФИКС ОШИБКИ: Проверка наличия Rust перед началом
    if not HAS_RUST: 
        return "❌ Error: 'bulba_rust' library is missing! Reinstall app.", None, 0, pd.DataFrame(), "Error"
    
    gc.collect()
    files = []
    
    # ФИКС ОШИБКИ: Безопасная обработка путей
    try:
        search_path = local_path if local_path else DIRS["DATA"]
        if upload_files: 
            # Gradio может возвращать объекты или строки
            for f in upload_files:
                if isinstance(f, str): files.append(f)
                elif hasattr(f, 'name'): files.append(f.name)
        elif os.path.exists(search_path):
            if os.path.isfile(search_path): files = [search_path]
            else:
                for ext in ['.txt', '.json', '.md', '.csv', '.py']:
                    files.extend(glob.glob(os.path.join(search_path, "**", ext), recursive=True))
    except Exception as e:
        return f"❌ File Error: {str(e)}", None, 0, pd.DataFrame(), "Error"

    if not files: 
        return f"⚠️ No files found in {search_path}", TRAIN_BIN_PATH, 0, pd.DataFrame(), "Error"

    tok_dir = DIRS["TOKENIZER"]
    # Обучаем токенизатор если нет
    if not os.path.exists(os.path.join(tok_dir, "tokenizer.model")):
        progress(0.1, desc="Training SentencePiece...")
        try:
            train_sentencepiece_model(files, 32000, tok_dir)
        except Exception as e:
            return f"❌ Tokenizer Error: {e}", None, 0, pd.DataFrame(), "Error"

    progress(0.3, desc="Rust: Tokenizing...")
    if os.path.exists(TRAIN_BIN_PATH): os.remove(TRAIN_BIN_PATH)

    abs_files = [os.path.abspath(f) for f in files]
    abs_tok = os.path.abspath(os.path.join(tok_dir, "tokenizer.json"))
    abs_out = os.path.abspath(TRAIN_BIN_PATH)

    task = {"done": False, "tokens": 0, "error": None}
    
    def run_rust():
        try:
            # ФИКС: Вызов Rust в try-except
            res = bulba_rust.process_parallel(abs_files, abs_tok, abs_out)
            task["tokens"] = int(res)
        except Exception as e: 
            task["error"] = str(e)
            print(f"Rust Error details: {e}")
        finally: task["done"] = True

    t = threading.Thread(target=run_rust)
    t.start()
    
    hist_data = []
    start_time = time.time()
    
    while t.is_alive():
        time.sleep(0.5)
        elapsed = time.time() - start_time
        if os.path.exists(TRAIN_BIN_PATH):
            cur_sz = os.path.getsize(TRAIN_BIN_PATH)
            avg_speed = (cur_sz / (1024*1024)) / (elapsed + 0.001)
            hist_data.append({"time": elapsed, "speed": avg_speed})
            df = pd.DataFrame(hist_data[-100:] if len(hist_data)>100 else hist_data)
            yield f"Processing... {format_size(cur_sz)}", TRAIN_BIN_PATH, 0, df, "Working..."
        
    t.join()

    if task["error"]:
        return f"❌ Rust Failed: {task['error']}", None, 0, pd.DataFrame(), "Error"

    if task["done"] and os.path.exists(TRAIN_BIN_PATH):
        try:
            progress(0.9, desc="Shuffling...")
            yield "🎲 Shuffling dataset...", TRAIN_BIN_PATH, 0, pd.DataFrame(hist_data), "Shuffling..."
            
            eos_id = 2 
            temp_bin = os.path.join(DIRS["DATA"], "temp_unshuffled.bin")
            if os.path.exists(temp_bin): os.remove(temp_bin)
            os.rename(TRAIN_BIN_PATH, temp_bin)
            
            bulba_rust.shuffle_dataset(temp_bin, TRAIN_BIN_PATH, eos_id)
            if os.path.exists(temp_bin): os.remove(temp_bin)
        except Exception as e:
            print(f"❌ Shuffle Failed: {e}")
            if os.path.exists(temp_bin): os.rename(temp_bin, TRAIN_BIN_PATH)

    final_sz = os.path.getsize(TRAIN_BIN_PATH) if os.path.exists(TRAIN_BIN_PATH) else 0
    return f"✅ DONE! Tokens: {task['tokens']:,} | Size: {format_size(final_sz)}", TRAIN_BIN_PATH, 0, pd.DataFrame(hist_data), "Done"

# ================= 🧠 TRAINING LOOP =================

STOP_FLAG = False
SAVE_FLAG = False 

def save_safe(model, path, optim=None, step=0):
    folder = os.path.dirname(path)
    os.makedirs(folder, exist_ok=True)
    sd = model.state_dict()
    save_file(sd, path)
    if optim:
        optim_path = path.replace(".safetensors", ".pt")
        torch.save(optim.state_dict(), optim_path)

def train_loop(train_path, model_name, lr_in, steps_ui, mode, resume_path, use_logging, use_neftune, progress=gr.Progress()):
    global STOP_FLAG, SAVE_FLAG
    STOP_FLAG = False
    SAVE_FLAG = False
    
    gc.collect()
    torch.cuda.empty_cache() if DEVICE == "cuda" else None
    
    writer = None
    if use_logging and HAS_TB:
        run_name = f"{model_name.split()[0]}-{datetime.datetime.now().strftime('%d_%H-%M')}"
        writer = SummaryWriter(log_dir=os.path.join(DIRS["LOGS"], run_name))
    
    target_bin = train_path if train_path else TRAIN_BIN_PATH
    if not os.path.exists(target_bin): return f"❌ Data not found", None, pd.DataFrame()
    
    try: tok = LlamaTokenizerFast.from_pretrained(DIRS["TOKENIZER"])
    except: return "❌ Tokenizer Error", None, pd.DataFrame()

    p = PRESETS[model_name]
    max_steps = int(steps_ui)
    
    conf = LlamaConfig(
        vocab_size=len(tok), hidden_size=p["h"], intermediate_size=p["i"],
        num_hidden_layers=p["l"], num_attention_heads=p["hd"], num_key_value_heads=p["kv"],
        max_position_embeddings=p["ctx"]
    )
    
    # ФИКС: Улучшена поддержка Mac (MPS использует float32 для стабильности)
    dtype_load = torch.bfloat16 if (DEVICE == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
    model = LlamaForCausalLM(conf).to(device=DEVICE, dtype=dtype_load)
    
    start_step = 0
    if resume_path and os.path.exists(resume_path):
        try:
            model.load_state_dict(load_file(resume_path), strict=False)
            match = re.search(r"(\d+)", os.path.basename(resume_path))
            if match: start_step = int(match.group(1))
        except Exception as e: return f"❌ Resume Error: {e}", None, pd.DataFrame()

    if use_neftune:
        def neftune_hook(module, input, output):
            if module.training:
                output = output + torch.zeros_like(output).uniform_(-5.0/torch.sqrt(torch.tensor(output.size(1)*output.size(2))), 5.0)
            return output
        model.get_input_embeddings().register_forward_hook(neftune_hook)

    # Компиляция только если не Mac и выбрано
    if not IS_MAC and ("Compile" in str(mode)) and DEVICE == "cuda":
        try: model = torch.compile(model)
        except: pass

    model.train()
    optim = torch.optim.AdamW(model.parameters(), lr=lr_in)
    
    if start_step > 0:
        op_path = resume_path.replace(".safetensors", ".pt")
        if os.path.exists(op_path):
            try: optim.load_state_dict(torch.load(op_path, map_location=DEVICE))
            except: pass

    try: loader = bulba_rust.RustDataLoader(target_bin, tok.eos_token_id, True)
    except: return "❌ Loader Error (Rust missing?)", None, pd.DataFrame()

    hist_tr = []
    avg_loss = 0
    t0 = time.time()
    
    for step in range(start_step, max_steps):
        if STOP_FLAG: break
        
        current_lr = get_lr(step, max_steps, lr_in)
        for g in optim.param_groups: g['lr'] = current_lr
        
        optim.zero_grad(set_to_none=True) 
        loss_accum = 0
        
        for _ in range(p["acc"]):
            raw = loader.next_batch(p["bs"], p["ctx"] + 1)
            data_np = np.frombuffer(raw, dtype=np.uint16).astype(np.int64)
            data = torch.from_numpy(data_np).view(p["bs"], p["ctx"] + 1).to(DEVICE, non_blocking=True)
            data.clamp_(max=len(tok)-1)

            loss = model(data[:, :-1], labels=data[:, 1:]).loss / p["acc"]
            loss.backward()
            loss_accum += loss.item()
        
        optim.step()
        
        if step == start_step: avg_loss = loss_accum
        else: avg_loss = 0.95 * avg_loss + 0.05 * loss_accum
        
        if writer and step % 10 == 0: writer.add_scalar("Train/Loss", avg_loss, step)
        
        if SAVE_FLAG:
            save_safe(model, f"{DIRS['CHECKPOINTS']}/manual_{step}.safetensors", optim)
            SAVE_FLAG = False

        if step % 10 == 0:
            dt = time.time() - t0
            spd = int((step - start_step + 1) * p["bs"] * p["acc"] * p["ctx"] / (dt + 0.001))
            eta = str(datetime.timedelta(seconds=int((max_steps - step) / ((step - start_step + 1)/dt))))
            hist_tr.append({"step": step, "loss": avg_loss})
            yield f"Step: {step} | Loss: {avg_loss:.4f} | Speed: {spd} tok/s | ETA: {eta}", pd.DataFrame(hist_tr)
            
            if step % 500 == 0 and step > 0:
                save_safe(model, f"{DIRS['CHECKPOINTS']}/step_{step}.safetensors", optim)

    save_safe(model, f"{DIRS['CHECKPOINTS']}/final.safetensors", optim)
    yield "✅ Done!", pd.DataFrame(hist_tr)

def stop_train(): global STOP_FLAG; STOP_FLAG = True; return "🛑 Stopping..."
def trigger_save(): global SAVE_FLAG; SAVE_FLAG = True; return "💾 Saving..." 

# ================= 📦 EXPORT =================
def auto_export_gguf(selected_ckpt, model_name_ui):
    if not selected_ckpt: return "❌ No checkpoint selected"
    yield f"Processing {selected_ckpt}..."
    script_path = os.path.join(DIRS["TOOLS"], "convert_hf_to_gguf.py")
    if not os.path.exists(script_path):
        try:
            r = requests.get("https://raw.githubusercontent.com/ggerganov/llama.cpp/master/convert_hf_to_gguf.py")
            with open(script_path, "wb") as f: f.write(r.content)
        except: return "❌ Download Failed"

    hf_path = DIRS["TEMP"]
    shutil.rmtree(hf_path, ignore_errors=True)
    os.makedirs(hf_path, exist_ok=True)
    
    try:
        shutil.copy(os.path.join(DIRS["TOKENIZER"], "tokenizer.model"), os.path.join(hf_path, "tokenizer.model"))
        p = PRESETS.get(model_name_ui, PRESETS["AIst (250M)"])
        cfg = {
            "architectures": ["LlamaForCausalLM"], "model_type": "llama", "vocab_size": 32000,
            "hidden_size": p["h"], "intermediate_size": p["i"], "num_hidden_layers": p["l"],
            "num_attention_heads": p["hd"], "num_key_value_heads": p["kv"], "max_position_embeddings": p["ctx"],
            "rope_theta": 10000.0, "bos_token_id": 1, "eos_token_id": 2, "rms_norm_eps": 1e-5
        }
        with open(os.path.join(hf_path, "config.json"), "w") as f: json.dump(cfg, f, indent=2)
        shutil.copy(selected_ckpt, os.path.join(hf_path, "model.safetensors"))
        
        out_file = os.path.join(DIRS["MODELS"], f"bulba_{os.path.basename(selected_ckpt)}.gguf")
        cmd = [sys.executable, script_path, hf_path, "--outfile", out_file, "--outtype", "f16"]
        subprocess.run(cmd, capture_output=True)
        yield f"✅ Saved: {out_file}"
    except Exception as e: yield f"❌ Error: {e}"

# ================= UI =================
def update_params(name):
    p = PRESETS[name]
    return p["steps"], p["lr"]

with gr.Blocks(title="BulbaGPT Studio") as demo:
    gr.Markdown("# 🥔 BulbaGPT Studio")
    s_path = gr.State(TRAIN_BIN_PATH)
    s_data_dir = gr.State(DIRS["DATA"])

    with gr.Tabs():
        with gr.Tab("🚀 Train"):
            with gr.Row():
                with gr.Column():
                    files = gr.File(label="Files", file_count="multiple")
                    btn_proc = gr.Button("1. Prepare Data", variant="secondary")
                    status_proc = gr.Textbox(label="Status")
                    
                    model_sel = gr.Radio(list(PRESETS.keys()), value="AIst (250M)", label="Config")
                    with gr.Row():
                        steps = gr.Number(label="Steps", value=12000)
                        lr_input = gr.Number(label="LR", value=4e-4, precision=6)
                    
                    model_sel.change(update_params, inputs=model_sel, outputs=[steps, lr_input])
                    
                    # ФИКС ИНТЕРФЕЙСА: Скрываем настройки CUDA на Mac
                    with gr.Column(visible=not IS_MAC):
                        mode = gr.Dropdown(["Compile", "Flash Attention", "Both"], value="Flash Attention", label="NVIDIA Mode")
                    
                    # Если Mac, передаем заглушку
                    if IS_MAC:
                        mode = gr.State("Mac Default")

                    with gr.Accordion("Advanced", open=False):
                        use_logging = gr.Checkbox(label="TensorBoard", value=True)
                        use_neftune = gr.Checkbox(label="NEFTune", value=False)
                    
                    resume_ckpt = gr.Dropdown(label="Resume Checkpoint", choices=[], value=None)
                    
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
    demo.load(scan_checkpoints, outputs=resume_ckpt)
    
    btn_proc.click(process_data, [files, s_data_dir], [status_proc, s_path, gr.State(), gr.State(), gr.State()])
    btn_run.click(train_loop, [s_path, model_sel, lr_input, steps, mode, resume_ckpt, use_logging, use_neftune], [status_train, plot])
    btn_stop.click(stop_train, outputs=status_train)
    btn_save.click(trigger_save, outputs=status_train)
    btn_ref.click(scan_checkpoints, outputs=ckpt)
    btn_exp.click(auto_export_gguf, inputs=[ckpt, model_sel], outputs=log_exp)

if __name__ == "__main__":
    start_tensorboard_background()
    # Запускаем в браузере
    demo.queue().launch(inbrowser=True)