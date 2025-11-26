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
OUTPUT_FILE = os.path.join(DATA_DIR, "aist_150m_mixed.txt")

LOCAL_PURE_TEXT_FILE = os.path.join(DATA_DIR, "pure_bel_books.txt") 
LOCAL_CULTURAX_PATH = "/home/sehaxe/bulbagpt/data/CulturalX_bel"

# ⚙️ ЛІМІТЫ
UPSAMPLE_INSTRUCT = 5     
UPSAMPLE_PURE_TEXT = 3    
LIMIT_CULTURAX = 150_000  
LIMIT_PYTHON = 30_000     
LIMIT_LOGIC_EN = 70_000   

MIN_TEXT_LENGTH = 300     
BUFFER_SIZE = 20000       # 🔥 Сброс на диск каждые 20k документов (бережет RAM)

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

def format_llama3_instruct(system, user, assistant):
    sys = sanitize_text(system) or "Ты разумны і карысны памочнік."
    usr = sanitize_text(user)
    ast = sanitize_text(assistant)
    if not usr or not ast: return ""
    return (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{sys}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n{usr}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n{ast}<|eot_id|><|end_of_text|>\n"
    )

def format_pretrain(text):
    text = sanitize_text(text)
    if len(text) < MIN_TEXT_LENGTH: return ""
    return f"<|begin_of_text|>{text}<|end_of_text|>\n"

# ================= GENERATORS (STREAMING) =================
# Генераторы выдают по одной строчке за раз, не занимая память

def stream_books():
    """Читает файл книг чанками, не загружая целиком"""
    if not os.path.exists(LOCAL_PURE_TEXT_FILE): return
    print("📘 Init Books stream...")
    try:
        # Читаем построчно или блоками, накапливая абзацы
        current_chunk = ""
        with open(LOCAL_PURE_TEXT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip() == "": # Пустая строка - разделитель абзацев
                    if len(current_chunk) > MIN_TEXT_LENGTH:
                        formatted = format_pretrain(current_chunk)
                        for _ in range(UPSAMPLE_PURE_TEXT):
                            yield formatted
                        current_chunk = ""
                    else:
                        current_chunk += "\n" # Просто перенос, если мало текста
                else:
                    current_chunk += line
        # Последний кусок
        if len(current_chunk) > MIN_TEXT_LENGTH:
             formatted = format_pretrain(current_chunk)
             for _ in range(UPSAMPLE_PURE_TEXT): yield formatted
    except Exception as e: print(f"❌ Books Error: {e}")

def stream_alpaca():
    print("💬 Init Alpaca stream...")
    try:
        ds = load_dataset("saillab/alpaca-belarusian-cleaned", split="train", streaming=True)
        # Поскольку датасет маленький, можно прочитать его, но yield-ить по одному
        # Streaming mode для HuggingFace datasets не грузит RAM
        for row in ds:
            user_msg = f"{row.get('instruction','')} {row.get('input','')}"
            text = format_llama3_instruct("", user_msg, row.get('output',''))
            if text:
                for _ in range(UPSAMPLE_INSTRUCT): # Upsample "on the fly"
                    yield text
    except Exception as e: print(f"❌ Alpaca Error: {e}")

def stream_python():
    print("🐍 Init Python stream...")
    try:
        ds = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train", streaming=True)
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
    print("🧠 Init Wiki stream...")
    try:
        ds = load_dataset("wikimedia/wikipedia", "20231101.be", split="train", streaming=True)
        for row in ds:
            text = row.get('text', '')
            if is_pure_belarusian(text) and len(text) > MIN_TEXT_LENGTH:
                yield format_pretrain(text)
    except Exception as e: print(f"❌ Wiki Error: {e}")

def stream_cosmopedia():
    print("🇬🇧 Init Cosmopedia stream...")
    try:
        ds = load_dataset("HuggingFaceTB/cosmopedia", "stanford", split="train", streaming=True)
        count = 0
        for row in ds:
            if count >= LIMIT_LOGIC_EN: break
            yield format_pretrain(row['text'])
            count += 1
    except Exception as e: print(f"❌ Cosmopedia Error: {e}")

def stream_culturax():
    print("🌍 Init CulturaX stream...")
    try:
        parquet_files = glob.glob(os.path.join(LOCAL_CULTURAX_PATH, "*.parquet"))
        if not parquet_files: return
        random.shuffle(parquet_files)
        
        c_web = 0
        for p_file in parquet_files:
            if c_web >= LIMIT_CULTURAX: break
            try:
                # Читаем файл
                df = pd.read_parquet(p_file, columns=['text']).dropna()
                # Итерируемся по строкам
                for text in df['text']:
                    if c_web >= LIMIT_CULTURAX: break
                    if len(text) < MIN_TEXT_LENGTH: continue
                    if is_pure_belarusian(text):
                        yield format_pretrain(text)
                        c_web += 1
                
                # Чистим память после каждого файла
                del df
                gc.collect()
                
            except Exception as e: continue
    except Exception as e: print(f"❌ CulturaX Error: {e}")

# ================= MAIN MIXER =================

if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)

print(f"🚀 Starting STREAMING Generation...")
print(f"💾 Output: {OUTPUT_FILE}")

# 1. Создаем список активных генераторов
generators = [
    stream_books(),
    stream_alpaca(),
    stream_python(),
    stream_wiki(),
    stream_cosmopedia(),
    stream_culturax()
]

# Фильтруем пустые/упавшие генераторы сразу (пробный старт)
active_gens = []
for g in generators:
    if g is not None:
        active_gens.append(g)

buffer = []
total_written = 0

with open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
    pbar = tqdm(desc="Processing & Mixing", unit=" docs")
    
    while active_gens:
        # 1. Выбираем случайный источник (Random Selection)
        # Это обеспечивает перемешивание "на лету"
        gen_idx = random.randint(0, len(active_gens) - 1)
        current_gen = active_gens[gen_idx]
        
        try:
            # 2. Берем 1 документ
            doc = next(current_gen)
            buffer.append(doc)
            
            # 3. Если буфер полон -> Сброс на диск
            if len(buffer) >= BUFFER_SIZE:
                random.shuffle(buffer) # Перемешиваем внутри буфера
                for item in buffer:
                    f_out.write(item)
                total_written += len(buffer)
                pbar.update(len(buffer))
                buffer = [] # Очищаем RAM
                gc.collect()
                
        except StopIteration:
            # Источник закончился, удаляем из списка
            active_gens.pop(gen_idx)
        except Exception as e:
            print(f"⚠️ Stream Error: {e}")
            active_gens.pop(gen_idx)

    # 4. Записываем остатки буфера
    if buffer:
        random.shuffle(buffer)
        for item in buffer:
            f_out.write(item)
        total_written += len(buffer)
        pbar.update(len(buffer))

print(f"\n🎉 COMPLETE! Total Docs: {total_written}")
file_size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
print(f"💾 Final Size: {file_size_mb:.2f} MB")