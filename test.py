import torch
import os
import glob
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizerFast
from safetensors.torch import load_file

# ================= НАСТРОЙКИ =================
TOKENIZER_PATH = "system/tokenizer"
CHECKPOINT_DIR = "checkpoints"
MODEL_TYPE = "AIst (150M)" 

PRESETS = {
    "Krolik (50M)":  { "h": 512,  "i": 1376, "l": 8,  "hd": 8,  "kv": 4, "ctx": 512 },
    "AIst (150M)":   { "h": 768,  "i": 2048, "l": 12, "hd": 12, "kv": 4, "ctx": 512 },
    "Zubr (350M)":   { "h": 1024, "i": 2816, "l": 28, "hd": 16, "kv": 4, "ctx": 512 },
}
# =============================================

def get_latest_checkpoint():
    files = glob.glob(f"{CHECKPOINT_DIR}/*.safetensors")
    if not files: return None
    # Сортируем по времени изменения (самый новый - первый)
    return max(files, key=os.path.getmtime)

def clean_state_dict(sd):
    """Убирает префикс _orig_mod. (от torch.compile)"""
    new_sd = {}
    for k, v in sd.items():
        new_k = k.replace("_orig_mod.", "")
        new_sd[new_k] = v
    return new_sd

def generate_text():
    print(f"🔄 Загрузка токенизатора из {TOKENIZER_PATH}...")
    try:
        tokenizer = LlamaTokenizerFast.from_pretrained(TOKENIZER_PATH)
    except Exception as e:
        print(f"❌ Ошибка токенизатора: {e}")
        return

    ckpt_path = get_latest_checkpoint()
    if not ckpt_path:
        print("❌ Чекпоинты не найдены!")
        return
    print(f"📥 Выбран чекпоинт: {ckpt_path}")

    print(f"🏗️ Создание модели {MODEL_TYPE}...")
    p = PRESETS[MODEL_TYPE]
    
    config = LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=p["h"],
        intermediate_size=p["i"],
        num_hidden_layers=p["l"],
        num_attention_heads=p["hd"],
        num_key_value_heads=p["kv"],
        max_position_embeddings=p["ctx"],
        rope_theta=10000.0,
        attn_implementation="sdpa" # Ускоряет инференс
    )
    
    model = LlamaForCausalLM(config)
    
    print(f"💾 Загрузка весов...")
    state_dict = load_file(ckpt_path)
    state_dict = clean_state_dict(state_dict) # Чистим ключи
    
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"Статус загрузки: {msg}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Запуск на {device}")
    model.to(device)
    model.eval()

    while True:
        print("\n" + "="*40)
        prompt = input("📝 Увядзіце запыт (ці 'q' для выхаду): ")
        if prompt.lower() in ['q', 'exit']: break
        
        # Фармат Instruct (калі трэба)
        formatted_prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=100,
                temperature=0.6,
                top_k=40,
                repetition_penalty=1.15,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        result = tokenizer.decode(outputs[0], skip_special_tokens=False)
        # Трохі чысцім вывад, каб пакінуць толькі адказ
        answer = result.split("assistant<|end_header_id|>\n\n")[-1].replace("<|eot_id|>", "").replace("<|end_of_text|>", "")
        
        print(f"🤖 Адказ:\n{answer}")

if __name__ == "__main__":
    generate_text()