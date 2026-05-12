import os, sys, torch, librosa, numpy as np
import soundfile as sf
from peft import PeftModel
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

# -- Config --
MODEL_DIR = '/content/ttsfine/Qwen3-TTS-Base'
LORA_DIR = '/content/output_malayalam_lora/checkpoint-epoch-0' # Change to your latest epoch
OUTPUT_WAV = 'malayalam_output.wav'

def run_inference(text, ref_audio_path):
    print(f"Loading Base Model: {MODEL_DIR}")
    # Load base model in float16 to save VRAM
    qwen_tts = Qwen3TTSModel.from_pretrained(MODEL_DIR, torch_dtype=torch.float16)
    
    print(f"Applying LoRA Adapters from: {LORA_DIR}")
    # Apply LoRA to the internal model
    qwen_tts.model = PeftModel.from_pretrained(qwen_tts.model, LORA_DIR)
    qwen_tts.model.eval()
    qwen_tts.model.to("cuda")

    print(f"Generating Kerala voice for: {text}")
    # Generate audio
    # Note: ref_audio is required for the 'Base' model to define the speaker style
    wavs, sr = qwen_tts.generate_voice_clone(
        text=text,
        ref_audio=ref_audio_path,
        language="Malayalam"
    )

    # Save output
    sf.write(OUTPUT_WAV, wavs[0], sr)
    print(f"✅ Successfully saved to {OUTPUT_WAV}")

if __name__ == "__main__":
    # Example Malaylam text: "എന്റെ പേര് ഖ്വെൻ. നിങ്ങളെ കാണുന്നതിൽ സന്തോഷമുണ്ട്."
    # (My name is Qwen. Nice to meet you.)
    test_text = "എന്റെ പേര് ഖ്വെൻ. നിങ്ങളെ കാണുന്നതിൽ സന്തോഷമുണ്ട്."
    
    # You need any reference audio from your dataset folder
    # We'll try to pick the first one from your prepared data if it exists
    ref_audio = "/content/ttsfine/malayalam_data/sample_ref.wav" 
    
    if os.path.exists(ref_audio):
        run_inference(test_text, ref_audio)
    else:
        print(f"❌ Error: Please provide a valid reference audio path at {ref_audio}")
