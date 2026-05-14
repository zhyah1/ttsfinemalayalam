import os, sys, torch, librosa, numpy as np
import soundfile as sf
import io, re, logging
from peft import PeftModel
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from transformers import logging as transformers_logging

# 1. Silence Noisy Warnings
transformers_logging.set_verbosity_error()
logging.getLogger("transformers").setLevel(logging.ERROR)

# Fix Windows console encoding for Malayalam/Emojis
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# -- Config --
if os.name == 'nt':  # Windows
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_DIR = os.path.join(BASE_DIR, 'Qwen3-TTS-Base')
    LORA_DIR  = os.path.join(BASE_DIR, 'output-malayalam-lora')
    REF_AUDIO = r"C:\Users\siyah\Pictures\ttsfine\videoplayback (2).m4a"
else:  # Colab
    MODEL_DIR = '/content/ttsfine/Qwen3-TTS-Base'
    LORA_DIR  = '/content/output-malayalam-lora'
    REF_AUDIO = '/content/ttsfine/videoplayback (2).m4a'

OUTPUT_WAV = 'malayalam_output_v23.wav'

# Global model variable to avoid re-loading across function calls
_qwen_tts_instance = None

def get_model():
    """Singleton pattern to load the model only once."""
    global _qwen_tts_instance
    if _qwen_tts_instance is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"--- Initializing Olam TTS Engine ({device}). Please wait... ---")
        
        # Load Base Model (Fixed deprecated torch_dtype call)
        model = Qwen3TTSModel.from_pretrained(
            MODEL_DIR, 
            dtype=torch.float16 if device == "cuda" else torch.float32,
            attn_implementation="sdpa" if device == "cuda" else "eager"
        )
        
        if os.path.exists(LORA_DIR):
            print(f"Applying LoRA voice adapters from {LORA_DIR}...")
            model.model = PeftModel.from_pretrained(model.model, LORA_DIR)
        
        model.model.eval()
        model.model.to(device)
        model.device = torch.device(device)
        _qwen_tts_instance = model
    return _qwen_tts_instance

def clean_malayalam_text(text):
    """Keep only Malayalam characters and spaces, removing all punctuation/dots."""
    # Keep only Malayalam Unicode range (0D00-0D7F) and whitespace
    text = re.sub(r'[^\u0D00-\u0D7F\s]', ' ', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def split_sentences(text):
    """Split by actual spaces or large gaps to keep it stable."""
    # After cleaning, we split into smaller chunks of ~15 words for GPU/CPU stability
    words = text.split(' ')
    chunks = []
    current_chunk = []
    for word in words:
        current_chunk.append(word)
        if len(current_chunk) >= 15: # Optimal size for accurate generation
            chunks.append(' '.join(current_chunk))
            current_chunk = []
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks if chunks else [text]

def run_inference(text, ref_audio_path):
    # Get or initialize model
    qwen_tts = get_model()
    
    # 1. CLEAN: Remove All dots, symbols, and English characters
    cleaned_full_text = clean_malayalam_text(text)
    
    # 2. CHUNK: Break into stable pieces for the model
    chunks = split_sentences(cleaned_full_text)
    print(f"Synthesizing {len(chunks)} cleaned chunk(s)...")
    
    all_wavs = []
    sample_rate = 24000

    for i, s in enumerate(chunks):
        print(f"  > [{i+1}/{len(chunks)}] Talking: {s[:50]}...")
        wavs, sr = qwen_tts.generate_voice_clone(
            text=s,
            ref_audio=ref_audio_path,
            language="Malayalam",
            x_vector_only_mode=True,
            temperature=0.4,           # Stable sampling
            top_p=0.8,                 # Precise selection
            repetition_penalty=1.2,    # Prevent looping
            max_new_tokens=512,        # Token limit
            non_streaming_mode=True,
            use_cache=True
        )
        all_wavs.append(wavs[0])
        sample_rate = sr

    # Combine segments with a natural 0.2s breath pause
    pause = np.zeros(int(sample_rate * 0.2))
    combined = np.concatenate([w for wav in all_wavs for w in (wav, pause)][:-1])

    sf.write(OUTPUT_WAV, combined, sample_rate)
    print(f"✅ Full audio generated: {os.path.abspath(OUTPUT_WAV)}")

if __name__ == "__main__":
    # Changed test word to 'നമസ്കാരം' (Namaskaram)
    test_text = "നമസ്കാരം"
    
    if os.path.exists(REF_AUDIO):
        run_inference(test_text.strip(), REF_AUDIO)
    else:
        print(f"Error: Reference audio not found at {REF_AUDIO}")


