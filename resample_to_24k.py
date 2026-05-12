import os
import argparse
import librosa
import soundfile as sf
from tqdm import tqdm

def resample_dir(input_dir, sample_rate=24000):
    """
    Resamples all .wav files in a directory to the target sample rate and mono.
    Replaces original files as the bash script does.
    """
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(".wav")]
    count = 0
    
    print(f"Checking {len(files)} files in {input_dir}...")
    
    for filename in tqdm(files):
        filepath = os.path.join(input_dir, filename)
        try:
            # Load metadata to check sample rate without loading full audio
            sr = librosa.get_samplerate(filepath)
            
            if sr != sample_rate:
                # Load, resample to mono and target SR
                y, _ = librosa.load(filepath, sr=sample_rate, mono=True)
                # Overwrite original
                sf.write(filepath, y, sample_rate)
                count += 1
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print(f"Resampled {count} files to {sample_rate}Hz in {input_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resample WAV files to 24kHz Mono")
    parser.add_argument("audio_dir", help="Directory containing WAV files")
    parser.add_argument("--sr", type=int, default=24000, help="Target sample rate (default 24000)")
    
    args = parser.parse_args()
    resample_dir(args.audio_dir, args.sr)
