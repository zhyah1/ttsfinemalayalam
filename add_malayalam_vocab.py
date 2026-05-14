"""
add_malayalam_vocab.py
======================
Expands the Qwen3-TTS-Base tokenizer to include every Malayalam Unicode
character (U+0D00–U+0D7F) as first-class vocabulary tokens, and adds
Malayalam to the talker codec_language_id table in config.json.

Run ONCE before any fine-tuning:
    python add_malayalam_vocab.py --model_dir Qwen3-TTS-Base

After this script the model directory will be ready for Malayalam fine-tuning.
The training script must resize the text_embedding layer to match the new
vocabulary size (handled automatically by resize_token_embeddings in training).
"""

import argparse
import json
import os
import shutil
from pathlib import Path


# ── Malayalam Unicode block ──────────────────────────────────────────────────
MALAYALAM_RANGE = range(0x0D00, 0x0D80)          # 128 code points

# Extra bigrams / common conjuncts frequently seen in Malayalam text
# (the BPE trainer would discover these; we add them explicitly so padding
#  the charset alone is not the only gain)
MALAYALAM_COMMON_BIGRAMS = [
    "ക്ക", "ന്ന", "ത്ത", "പ്പ", "ല്ല", "മ്മ", "ക്ക", " പ്ര",
    "ക്ഷ", "ത്ര", "ത്ത", "ന്ത", "ണ്ണ", "ണ്ട", "ങ്ക", "ന്ദ",
    "ഞ്ഞ", "ഞ്ച", "ര്‍", "ണ്‍", "ള്‍", "ന്‍", "ല്‍",
]


def get_malayalam_characters():
    """Return a list of printable Malayalam Unicode characters."""
    chars = []
    for cp in MALAYALAM_RANGE:
        c = chr(cp)
        # Skip control / unassigned code points (they have no name)
        try:
            import unicodedata
            unicodedata.name(c)
            chars.append(c)
        except ValueError:
            pass
    return chars


def expand_tokenizer(model_dir: str, dry_run: bool = False):
    model_dir = Path(model_dir)
    vocab_file   = model_dir / "vocab.json"
    merges_file  = model_dir / "merges.txt"
    tok_cfg_file = model_dir / "tokenizer_config.json"
    cfg_file     = model_dir / "config.json"

    # ── Sanity checks ────────────────────────────────────────────────────────
    for f in [vocab_file, merges_file, tok_cfg_file, cfg_file]:
        if not f.exists():
            raise FileNotFoundError(f"Required file not found: {f}")

    # ── Load existing vocab ──────────────────────────────────────────────────
    print("Loading vocab.json …")
    with open(vocab_file, encoding="utf-8") as fh:
        vocab: dict = json.load(fh)

    existing_tokens = set(vocab.keys())
    next_id = max(vocab.values()) + 1
    print(f"  Current vocab size : {len(vocab):,}")
    print(f"  Next available id  : {next_id}")

    # ── Collect new Malayalam tokens ─────────────────────────────────────────
    ml_chars = get_malayalam_characters()
    all_new_tokens = ml_chars + [
        b for b in MALAYALAM_COMMON_BIGRAMS if b not in existing_tokens
    ]
    new_tokens = [t for t in all_new_tokens if t not in existing_tokens]

    if not new_tokens:
        print("No new Malayalam tokens to add — vocab already expanded.")
    else:
        print(f"  New Malayalam tokens: {len(new_tokens)}")
        for tok in new_tokens:
            vocab[tok] = next_id
            next_id += 1
        print(f"  Updated vocab size : {len(vocab):,}")

    # ── Update tokenizer_config.json ─────────────────────────────────────────
    print("Loading tokenizer_config.json …")
    with open(tok_cfg_file, encoding="utf-8") as fh:
        tok_cfg: dict = json.load(fh)

    # added_tokens_decoder maps string-id -> token-object
    adt: dict = tok_cfg.get("added_tokens_decoder", {})
    # additional_special_tokens list
    ast: list = tok_cfg.get("additional_special_tokens", [])

    added_count = 0
    for tok in new_tokens:
        tok_id = vocab[tok]
        str_id = str(tok_id)
        if str_id not in adt:
            adt[str_id] = {
                "content": tok,
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": False,
            }
            added_count += 1
        if tok not in ast:
            # We do NOT add individual glyphs as special tokens —
            # that would prevent the model from attending to them as ordinary
            # text. We just register them in added_tokens_decoder for
            # deterministic id-↔-string mapping.
            pass

    tok_cfg["added_tokens_decoder"] = adt
    print(f"  added_tokens_decoder updated with {added_count} new entries")

    # ── Update config.json — add Malayalam language id ───────────────────────
    print("Loading config.json …")
    with open(cfg_file, encoding="utf-8") as fh:
        cfg: dict = json.load(fh)

    talker_cfg = cfg.get("talker_config", {})
    codec_lang_id: dict = talker_cfg.get("codec_language_id", {})

    MALAYALAM_CODEC_ID = 2072   # next after Portuguese (2071)

    if "malayalam" not in codec_lang_id:
        codec_lang_id["malayalam"] = MALAYALAM_CODEC_ID
        print(f"  Added  'malayalam' -> codec_language_id={MALAYALAM_CODEC_ID}")
    else:
        print(f"  'malayalam' already in codec_language_id "
              f"(id={codec_lang_id['malayalam']})")

    talker_cfg["codec_language_id"] = codec_lang_id
    # Also update text_vocab_size to match the new vocab
    new_text_vocab_size = len(vocab)
    old_text_vocab_size = talker_cfg.get("text_vocab_size", 0)
    talker_cfg["text_vocab_size"] = new_text_vocab_size
    if old_text_vocab_size != new_text_vocab_size:
        print(f"  text_vocab_size: {old_text_vocab_size} -> {new_text_vocab_size}")

    cfg["talker_config"] = talker_cfg

    # ── Write updated files (with backup) ────────────────────────────────────
    if dry_run:
        print("\n[DRY RUN] No files written.")
        return

    # Backup originals
    backup_dir = model_dir / "_backup_pre_malayalam"
    backup_dir.mkdir(exist_ok=True)
    for src in [vocab_file, tok_cfg_file, cfg_file]:
        dst = backup_dir / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
            print(f"  Backed up {src.name} -> {backup_dir.name}/")

    with open(vocab_file, "w", encoding="utf-8") as fh:
        json.dump(vocab, fh, ensure_ascii=False, indent=None, separators=(",", ":"))
    print(f"  ✓ Saved {vocab_file}")

    with open(tok_cfg_file, "w", encoding="utf-8") as fh:
        json.dump(tok_cfg, fh, ensure_ascii=False, indent=2)
    print(f"  ✓ Saved {tok_cfg_file}")

    with open(cfg_file, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, ensure_ascii=False, indent=2)
    print(f"  ✓ Saved {cfg_file}")

    print(f"\n✅ Done! New vocab size = {len(vocab):,} tokens")
    print(f"   Malayalam codec language id = {MALAYALAM_CODEC_ID}")
    print(f"\n⚠  IMPORTANT: After loading the model for training, call:")
    print(f"     model.resize_token_embeddings(len(new_vocab))")
    print(f"   to extend the text_embedding weight matrix.")
    print(f"   The training script handles this automatically.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Add Malayalam vocab tokens and language id to Qwen3-TTS-Base"
    )
    parser.add_argument(
        "--model_dir", type=str, default="Qwen3-TTS-Base",
        help="Path to the local model directory (default: Qwen3-TTS-Base)"
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print what would change without writing any files"
    )
    args = parser.parse_args()
    expand_tokenizer(args.model_dir, dry_run=args.dry_run)
