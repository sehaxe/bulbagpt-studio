import os
import struct
from transformers import LlamaTokenizerFast

# Пути
BIN_PATH = "data/train.bin"
TOK_PATH = "system/tokenizer"

def inspect():
    if not os.path.exists(BIN_PATH):
        print("❌ Нет файла train.bin")
        return

    print(f"🔍 Читаем {BIN_PATH}...")
    
    # 1. Загружаем токенизатор
    try:
        tokenizer = LlamaTokenizerFast.from_pretrained(TOK_PATH)
        print(f"✅ Токенизатор загружен. EOS ID: {tokenizer.eos_token_id}")
    except:
        print("❌ Токенизатор не найден!")
        return

    # 2. Читаем первые 200 токенов (чисел) из бинарника
    with open(BIN_PATH, "rb") as f:
        # Читаем 400 байт (так как uint16 = 2 байта)
        raw_data = f.read(400)
        # Распаковываем в числа
        tokens = struct.unpack(f"{len(raw_data)//2}H", raw_data)

    print(f"\n🔢 Первые 20 токенов (ID): {tokens[:20]}")
    
    # 3. Декодируем обратно в текст
    decoded_text = tokenizer.decode(tokens)
    
    print("\n📜 ВОТ ЧТО ВИДИТ НЕЙРОСЕТЬ (Первые 200 токенов):")
    print("="*40)
    print(decoded_text)
    print("="*40)

if __name__ == "__main__":
    inspect()