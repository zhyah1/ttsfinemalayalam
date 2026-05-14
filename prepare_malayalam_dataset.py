import os
os.environ["PATH"] = r"C:\Users\siyah\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.1-full_build\bin" + os.pathsep + os.environ["PATH"]

import json
import re
import argparse
import torch
import soundfile as sf
import librosa
import io
from datasets import load_dataset, Audio

from tqdm import tqdm
from qwen_tts import Qwen3TTSTokenizer


def clean_malayalam_text(text: str) -> str:
    """
    Light normalisation for Malayalam TTS text:
    - Strips ASCII punctuation that confuse the LLM (but keeps Malayalam punctuation).
    - Collapses multiple spaces.
    - Does NOT transliterate — raw Malayalam script is used directly.
    """
    # Remove ASCII symbols that are not meaningful phonetically
    text = re.sub(r'[\x00-\x1F\x7F]', '', text)          # control chars
    text = re.sub(r'["\[\]{}|<>]', '', text)              # stray ASCII brackets
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def prepare_dataset(dataset_name, output_dir, tokenizer_path, split_ratio=0.9, batch_size=32):
    # 1. Setup
    print(f"--- Preparing dataset: {dataset_name} ---")
    audio_dir = os.path.join(output_dir, "wavs")
    os.makedirs(audio_dir, exist_ok=True)
    
    # 2. Tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Qwen3-TTS Tokenizer on {device}...")
    tokenizer = Qwen3TTSTokenizer.from_pretrained(tokenizer_path, device_map=device)
    
    # 3. Load HF Dataset
    print(f"Downloading dataset '{dataset_name}'...")
    ds = load_dataset(dataset_name, split="train")
    
    # Disable automatic decoding to avoid broken torchcodec/ffmpeg dependencies in datasets
    print("Disabling automatic audio decoding...")
    ds = ds.cast_column("audio", Audio(decode=False))
    
    # 4. Filter and Process


    processed_items = []
    
    print(f"Decoding audios, resampling, and calculating codec tokens (Batch Size: {batch_size})...")
    
    batch_audios = []
    batch_metadata = []
    
    TARGET_SR = 24000
    
    for i, item in enumerate(tqdm(ds)):
        try:
            # item['audio'] contains {'path': ..., 'bytes': ...} when decode=False
            audio_info = item['audio']
            text = item['text']
            filename = item.get('filename', f"sample_{i:04d}.wav")
            
            # Decode bytes manually
            audio_bytes = audio_info.get('bytes')
            if audio_bytes is None:
                # Fallback if bytes aren't present (rare in modern HF datasets unless local)
                print(f"Skipping {filename}: No audio bytes found.")
                continue
                
            waveform, orig_sr = sf.read(io.BytesIO(audio_bytes))
            
            # Resample to 24kHz Mono if needed
            if orig_sr != TARGET_SR:
                waveform = librosa.resample(waveform, orig_sr=orig_sr, target_sr=TARGET_SR)
            
            # Ensure mono
            if waveform.ndim > 1:
                waveform = waveform.mean(axis=-1)
                
            # Save local wav
            wav_path = os.path.join(audio_dir, filename)
            sf.write(wav_path, waveform, TARGET_SR)
            
            text = clean_malayalam_text(text)

            metadata = {
                "audio": os.path.abspath(wav_path),
                "text": text,
                "ref_audio": os.path.abspath(wav_path),
                "language": "Malayalam",
            }
            
            batch_audios.append(wav_path)
            batch_metadata.append(metadata)
            
            # Process in batches for speed
            if len(batch_audios) >= batch_size:
                enc_res = tokenizer.encode(batch_audios)
                for code, meta in zip(enc_res.audio_codes, batch_metadata):
                    meta['audio_codes'] = code.cpu().tolist()
                    processed_items.append(meta)
                batch_audios.clear()
                batch_metadata.clear()
        except Exception as e:
            print(f"Error processing entry {i}: {e}")
            
    # Process remaining
    if batch_audios:
        try:
            enc_res = tokenizer.encode(batch_audios)
            for code, meta in zip(enc_res.audio_codes, batch_metadata):
                meta['audio_codes'] = code.cpu().tolist()
                processed_items.append(meta)
        except Exception as e:
            print(f"Error processing final batch: {e}")

    # 5. Split and Save
    if not processed_items:
        print("No items were successfully processed.")
        return

    print(f"Splitting dataset (ratio: {split_ratio})...")
    num_train = int(len(processed_items) * split_ratio)
    train_items = processed_items[:num_train]
    val_items = processed_items[num_train:]
    
    train_file = os.path.join(output_dir, "train_with_codes.jsonl")
    val_file = os.path.join(output_dir, "val_with_codes.jsonl")
    
    with open(train_file, 'w', encoding='utf-8') as f:
        for item in train_items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    with open(val_file, 'w', encoding='utf-8') as f:
        for item in val_items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    print(f"Done! Train: {len(train_items)}, Val: {len(val_items)}")
    print(f"Files saved in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="siyah1/Malayalam-TTS-v2")
    parser.add_argument("--output_dir", type=str, default="malayalam_data")
    parser.add_argument("--tokenizer_path", type=str, default="Qwen/Qwen3-TTS-Tokenizer-12Hz")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--split_ratio", type=float, default=0.9,
                        help="Fraction of data used for training (rest = validation)")
    args = parser.parse_args()

    prepare_dataset(
        args.dataset_name,
        args.output_dir,
        args.tokenizer_path,
        split_ratio=args.split_ratio,
        batch_size=args.batch_size,
    )
