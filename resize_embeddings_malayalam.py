"""
resize_embeddings_malayalam.py
==============================
After running add_malayalam_vocab.py (which updates vocab.json / config.json),
run this script ONCE to:

  1. Load Qwen3-TTS-Base
  2. Resize model.talker.model.text_embedding to cover the new Malayalam tokens
     (new rows are initialized by averaging existing rows — much better than
      random init, especially for byte-level fallback tokens that are similar
      to the Malayalam chars).
  3. Save the updated model weights back to --model_dir.

Usage:
    python resize_embeddings_malayalam.py --model_dir Qwen3-TTS-Base
"""

import argparse
import json
from pathlib import Path

import torch


def mean_pool_init(old_weight: torch.Tensor, new_size: int) -> torch.Tensor:
    """
    Extend an embedding matrix from old_size to new_size rows.
    New rows are initialised with the mean of existing rows ± small noise
    so gradients start flowing immediately.
    """
    old_size, hidden = old_weight.shape
    if new_size <= old_size:
        return old_weight

    mean_vec = old_weight.mean(dim=0, keepdim=True)      # (1, H)
    extra    = new_size - old_size
    noise    = torch.zeros(extra, hidden, dtype=old_weight.dtype)
    noise.normal_(std=0.01)
    new_rows = mean_vec.expand(extra, -1) + noise         # (extra, H)
    return torch.cat([old_weight, new_rows], dim=0)       # (new_size, H)


def resize_model(model_dir: str):
    import sys, os
    # Make sure the local qwen_tts package is importable
    repo_dir = Path(model_dir).parent / "Qwen3-TTS"
    if repo_dir.exists():
        sys.path.insert(0, str(repo_dir))

    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

    model_dir = Path(model_dir)

    # ── Load updated vocab size from config.json ──────────────────────────
    with open(model_dir / "config.json", encoding="utf-8") as fh:
        cfg = json.load(fh)
    new_text_vocab_size = cfg["talker_config"]["text_vocab_size"]
    print(f"Target text_vocab_size: {new_text_vocab_size:,}")

    # ── Load model (CPU to keep VRAM free) ───────────────────────────────
    print("Loading model (CPU, bfloat16) …")
    wrapper = Qwen3TTSModel.from_pretrained(
        str(model_dir),
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    model = wrapper.model   # Qwen3TTSForConditionalGeneration

    talker = model.talker   # the LM backbone

    # ── Locate text_embedding ─────────────────────────────────────────────
    # Depending on the exact model code it may be at:
    #   talker.model.text_embedding   (nn.Embedding)
    # or via talker.get_input_embeddings()
    emb_layer = None
    if hasattr(talker, "model") and hasattr(talker.model, "text_embedding"):
        emb_layer = talker.model.text_embedding
        emb_path  = "talker.model.text_embedding"
    elif hasattr(talker, "text_embedding"):
        emb_layer = talker.text_embedding
        emb_path  = "talker.text_embedding"
    else:
        # Fallback: use HF resize_token_embeddings which resizes input+output
        print("Could not locate text_embedding directly; using HF resize …")
        talker.resize_token_embeddings(new_text_vocab_size)
        print("  Done via resize_token_embeddings.")
        _save(model_dir, wrapper)
        return

    old_num, hidden = emb_layer.weight.shape
    print(f"Current embedding shape: ({old_num:,}, {hidden})")

    if old_num >= new_text_vocab_size:
        print(f"Embedding already large enough ({old_num:,} ≥ {new_text_vocab_size:,}). Nothing to do.")
        return

    # ── Build new weight ──────────────────────────────────────────────────
    print(f"Extending embedding {old_num:,} → {new_text_vocab_size:,} rows …")
    new_weight = mean_pool_init(emb_layer.weight.data, new_text_vocab_size)
    print(f"  New shape: {new_weight.shape}")

    # Replace the layer in-place
    import torch.nn as nn
    new_emb = nn.Embedding(new_text_vocab_size, hidden,
                           padding_idx=emb_layer.padding_idx,
                           dtype=emb_layer.weight.dtype)
    new_emb.weight = nn.Parameter(new_weight)
    # Walk the attribute path and replace
    parts = emb_path.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_emb)
    print(f"  ✓ Replaced {emb_path}")

    # ── Also update the config so safetensors save picks it up ───────────
    cfg["talker_config"]["text_vocab_size"] = new_text_vocab_size
    with open(model_dir / "config.json", "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, ensure_ascii=False, indent=2)
    print("  ✓ config.json updated")

    _save(model_dir, wrapper)


def _save(model_dir, wrapper):
    print("Saving updated model weights …")
    import shutil
    backup = Path(model_dir) / "_backup_pre_resize"
    backup.mkdir(exist_ok=True)
    sf = Path(model_dir) / "model.safetensors"
    bsf = backup / "model.safetensors"
    if sf.exists() and not bsf.exists():
        shutil.copy2(sf, bsf)
        print(f"  Backed up model.safetensors → {backup.name}/")

    wrapper.model.save_pretrained(str(model_dir), safe_serialization=True)
    print(f"  ✓ Saved to {model_dir}")
    print("\n✅ Embedding resize complete! You can now run fine-tuning.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="Qwen3-TTS-Base")
    args = parser.parse_args()
    resize_model(args.model_dir)
