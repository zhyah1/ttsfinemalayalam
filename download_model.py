import os
os.environ["HF_HUB_DISABLE_XET"] = "1"
from huggingface_hub import snapshot_download


repo_id = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
local_dir = "Qwen3-TTS-Base"

print(f"Starting download of {repo_id} to {local_dir}...")
try:
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        max_workers=1, # Sequential for stability
        resume_download=True,
    )
    print("Download complete!")
except Exception as e:
    print(f"Download failed: {e}")
