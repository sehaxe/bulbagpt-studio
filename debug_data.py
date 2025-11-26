import struct
from transformers import LlamaTokenizerFast

tokenizer = LlamaTokenizerFast.from_pretrained("system/tokenizer")

print("🔍 Читаем случайные куски из train.bin...")

with open("data/train.bin", "rb") as f:
    # Прыгаем в случайное место (например, на 10-й мегабайт)
    f.seek(10 * 1024 * 1024) 
    
    # Читаем 500 токенов
    raw = f.read(1000) 
    ids = struct.unpack("H" * 500, raw)
    
    text = tokenizer.decode(ids)
    print("-" * 50)
    print(text)
    print("-" * 50)