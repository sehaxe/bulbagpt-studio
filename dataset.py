import os
import re
import glob
import random
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
import gc

# ================= CONFIGURATION =================

DATA_DIR = "data"
OUTPUT_FILE = os.path.join(DATA_DIR, "aist_mixed.txt")

# Проверь свои пути!
LOCAL_PURE_TEXT_FILE = os.path.join(DATA_DIR, "pure_bel_books.txt") 
LOCAL_CULTURAX_PATH = "/home/sehaxe/bulbagpt/data/CulturalX_bel"

# ⚙️ ЛІМІТЫ
UPSAMPLE_INSTRUCT = 5     
UPSAMPLE_PURE_TEXT = 3    
LIMIT_CULTURAX = 150_000  
LIMIT_PYTHON = 30_000     
LIMIT_LOGIC_EN = 70_000   

MIN_TEXT_LENGTH = 300     
BUFFER_SIZE = 20000       

# ================= ФУНКЦЫІ АЧЫСТКІ =================

def sanitize_text(text):
    if not text: return ""
    text = re.sub(r"<\|im_start\|>.*?\n", "", text)
    text = text.replace("<|im_end|>", "").replace("<|endoftext|>", "")
    text = text.replace("<|begin_of_text|>", "")
    text = text.replace("<|start_header_id|>", "")
    text = text.replace("<|end_header_id|>", "")
    text = text.replace("<|eot_id|>", "")
    text = text.replace("<|end_of_text|>", "")
    return text.strip()

def is_pure_belarusian(text):
    if not text: return False
    text_lower = text.lower()
    if not re.search(r'[ўі]', text_lower): return False
    total_chars = len(text)
    if total_chars < 50: return False 
    bad_chars = len(re.findall(r'[щъи]', text_lower))
    if (bad_chars / total_chars) > 0.01: return False
    return True

# 🔥 ФОРМАТ ДЛЯ BISON 1.1B / LM STUDIO
def format_llama3_instruct(system, user, assistant):
    sys = sanitize_text(system) or "Ты разумны памочнік."
    usr = sanitize_text(user)
    ast = sanitize_text(assistant)
    if not usr or not ast: return ""
    return f"<s>System: {sys}\n\n### User: {usr}\n\n### Assistant: {ast}</s>\n"

def format_pretrain(text):
    text = sanitize_text(text)
    if len(text) < MIN_TEXT_LENGTH: return ""
    return f"<s>{text}</s>\n"

# ================= GENERATORS (OPTIMIZED) =================

def stream_books():
    print(f"🔍 CHECKING FILE: {os.path.abspath(LOCAL_PURE_TEXT_FILE)}")
    
    if not os.path.exists(LOCAL_PURE_TEXT_FILE): 
        print(f"❌ FILE NOT FOUND: {LOCAL_PURE_TEXT_FILE}")
        print("   -> Make sure the file is inside the 'data' folder!")
        return

    file_size = os.path.getsize(LOCAL_PURE_TEXT_FILE)
    print(f"   ✅ File found! Size: {file_size / 1024:.2f} KB")
    
    if file_size == 0:
        print("   ⚠️ File is EMPTY!")
        return

    print("📘 Init Books stream...")
    try:
        current_chunk = ""
        count_yield = 0
        
        with open(LOCAL_PURE_TEXT_FILE, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                # Если строка пустая - считаем это концом абзаца
                if line.strip() == "": 
                    if len(current_chunk) > MIN_TEXT_LENGTH:
                        formatted = format_pretrain(current_chunk)
                        for _ in range(UPSAMPLE_PURE_TEXT):
                            yield formatted
                            count_yield += 1
                        current_chunk = ""
                    else:
                        # Если кусок слишком маленький, просто добавляем перенос
                        current_chunk += "\n" 
                else:
                    current_chunk += line
        
        # Остаток в конце файла
        if len(current_chunk) > MIN_TEXT_LENGTH:
             formatted = format_pretrain(current_chunk)
             for _ in range(UPSAMPLE_PURE_TEXT): 
                 yield formatted
                 count_yield += 1
                 
        print(f"   📊 Books Stream Finished. Yielded: {count_yield} docs.")
        
    except Exception as e: print(f"❌ Books Error: {e}")

def stream_alpaca():
    print("⏳ Downloading Alpaca (Small)...")
    try:
        # streaming=False скачает файл один раз и не будет тупить
        ds = load_dataset("saillab/alpaca-belarusian-cleaned", split="train", streaming=False)
        print("✅ Alpaca loaded!")
        for row in ds:
            user_msg = f"{row.get('instruction','')} {row.get('input','')}"
            text = format_llama3_instruct("", user_msg, row.get('output',''))
            if text:
                for _ in range(UPSAMPLE_INSTRUCT): yield text
    except Exception as e: print(f"❌ Alpaca Error: {e}")

def stream_python():
    print("⏳ Downloading Python Code (Small)...")
    try:
        ds = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train", streaming=False)
        print("✅ Python loaded!")
        count = 0
        for row in ds:
            if count >= LIMIT_PYTHON: break
            prompt = row.get('instruction', '') + "\n" + row.get('input', '')
            code = row.get('output', '')
            if len(code) > 20 and len(code) < 8000:
                yield format_llama3_instruct("You are a Python coding assistant.", prompt, code)
                count += 1
    except Exception as e: print(f"❌ Python Error: {e}")

def stream_wiki():
    print("⏳ Downloading Wiki (Medium)...")
    try:
        ds = load_dataset("wikimedia/wikipedia", "20231101.be", split="train", streaming=False)
        print("✅ Wiki loaded!")
        for row in ds:
            text = row.get('text', '')
            if is_pure_belarusian(text) and len(text) > MIN_TEXT_LENGTH:
                yield format_pretrain(text)
    except Exception as e: print(f"❌ Wiki Error: {e}")

def stream_cosmopedia():
    # Тут оставляем стриминг, так как датасет огромный (25 ГБ)
    print("🌍 Init Cosmopedia (Streaming)...")
    try:
        ds = load_dataset("HuggingFaceTB/cosmopedia", "stanford", split="train", streaming=True)
        count = 0
        for row in ds:
            if count >= LIMIT_LOGIC_EN: break
            yield format_pretrain(row['text'])
            count += 1
    except Exception as e: print(f"❌ Cosmopedia Error: {e}")

def stream_culturax():
    # Тут локальные файлы, стриминг не нужен, просто читаем
    print("🌍 Init CulturaX (Local Parquet)...")
    try:
        parquet_files = glob.glob(os.path.join(LOCAL_CULTURAX_PATH, "*.parquet"))
        if not parquet_files: return
        random.shuffle(parquet_files)
        
        c_web = 0
        for p_file in parquet_files:
            if c_web >= LIMIT_CULTURAX: break
            try:
                df = pd.read_parquet(p_file, columns=['text']).dropna()
                for text in df['text']:
                    if c_web >= LIMIT_CULTURAX: break
                    if len(text) < MIN_TEXT_LENGTH: continue
                    if is_pure_belarusian(text):
                        yield format_pretrain(text)
                        c_web += 1
                del df
                gc.collect()
            except Exception as e: continue
    except Exception as e: print(f"❌ CulturaX Error: {e}")

# ================= MAIN MIXER =================

if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)

print(f"🚀 Starting GENERATION for BISON 1.1B...")
print(f"💾 Output: {OUTPUT_FILE}")

# Сначала создаем список генераторов
# При этом функции запустятся до первого yield и начнут качать данные
generators = [
    stream_books(),
    stream_alpaca(),
    stream_python(),
    stream_wiki(),
    stream_cosmopedia(),
    stream_culturax()
]

active_gens = [g for g in generators if g is not None]

buffer = []
total_written = 0

with open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
    pbar = tqdm(desc="Mixing Docs", unit=" docs")
    
    while active_gens:
        gen_idx = random.randint(0, len(active_gens) - 1)
        current_gen = active_gens[gen_idx]
        
        try:
            doc = next(current_gen)
            buffer.append(doc)
            
            if len(buffer) >= BUFFER_SIZE:
                random.shuffle(buffer)
                for item in buffer:
                    f_out.write(item)
                total_written += len(buffer)
                pbar.update(len(buffer))
                buffer = []
                gc.collect()
                
        except StopIteration:
            active_gens.pop(gen_idx)
        except Exception as e:
            # print(f"⚠️ Stream skip: {e}") # Скрываем мелкие ошибки
            active_gens.pop(gen_idx)

    if buffer:
        random.shuffle(buffer)
        for item in buffer:
            f_out.write(item)
        total_written += len(buffer)
        pbar.update(len(buffer))

print(f"\n🎉 COMPLETE! Total Docs: {total_written}")
file_size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
print(f"💾 Final Size: {file_size_mb:.2f} MB")