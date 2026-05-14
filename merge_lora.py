import os
import torch
from peft import PeftModel
from transformers import AutoConfig, AutoModel, AutoProcessor
from qwen_tts.core.models import Qwen3TTSConfig, Qwen3TTSForConditionalGeneration, Qwen3TTSProcessor

# -- Config --
BASE_MODEL_DIR = "Qwen3-TTS-Base"
LORA_ADAPTER_DIR = "output-malayalam-lora"
MERGED_OUTPUT_DIR = "Qwen3-TTS-Malayalam-Merged"

def merge_and_save():
    print(f"Loading Base Model from: {BASE_MODEL_DIR}")
    
    # Register model types (needed for Qwen3-TTS)
    AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
    AutoModel.register(Qwen3TTSConfig, Qwen3TTSForConditionalGeneration)
    AutoProcessor.register(Qwen3TTSConfig, Qwen3TTSProcessor)
    
    # 1. Load Base Model
    base_model = Qwen3TTSForConditionalGeneration.from_pretrained(
        BASE_MODEL_DIR, 
        torch_dtype=torch.float16,
        device_map="cpu" # Merge on CPU to avoid VRAM issues
    )
    
    # 2. Load LoRA Adapters
    print(f"Loading LoRA Adapters from: {LORA_ADAPTER_DIR}")
    model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_DIR)
    
    # 3. Merge and Unload
    print("Merging adapters into base model...")
    merged_model = model.merge_and_unload()
    
    # 4. Save Merged Model
    print(f"Saving merged model to: {MERGED_OUTPUT_DIR}")
    os.makedirs(MERGED_OUTPUT_DIR, exist_ok=True)
    merged_model.save_pretrained(MERGED_OUTPUT_DIR)
    
    # 5. Save Processor (copying from base)
    print("Saving processor...")
    processor = AutoProcessor.from_pretrained(BASE_MODEL_DIR, fix_mistral_regex=True)
    processor.save_pretrained(MERGED_OUTPUT_DIR)
    
    # Copy other necessary files (README, etc.) if they exist
    import shutil
    for filename in ["generation_config.json", "preprocessor_config.json"]:
        src = os.path.join(BASE_MODEL_DIR, filename)
        if os.path.exists(src):
            shutil.copy(src, MERGED_OUTPUT_DIR)

    print(f"✅ Successfully merged and saved to {MERGED_OUTPUT_DIR}")

if __name__ == "__main__":
    merge_and_save()
