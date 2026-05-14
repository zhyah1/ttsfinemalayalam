"""
verify_malayalam_tokenizer.py
==============================
Quick sanity check — loads the updated Qwen3-TTS-Base tokenizer and confirms:
  1. Every Malayalam Unicode char (U+0D00-U+0D7F) maps to a single token.
  2. Sample Malayalam sentences tokenize correctly.
  3. Detokenization round-trips perfectly.

Usage:
    python verify_malayalam_tokenizer.py --model_dir Qwen3-TTS-Base
"""
import argparse
import sys
from pathlib import Path
import unicodedata

SAMPLE_TEXTS = [
    "നമസ്കാരം",                              # Hello
    "എനിക്ക് മലയാളം സംസാരിക്കാം",            # I can speak Malayalam
    "ഈ ഒരു ശബ്ദ സമ്പ്ലേഷൺ പദ്ധതി ആണ്",        # This is a TTS project
    "ഒരു ഗ്ലാസ് വെള്ളം കൊണ്ടുവരൂ",             # Bring a glass of water
]

MALAYALAM_RANGE = range(0x0D00, 0x0D80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="Qwen3-TTS-Base")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)

    # Try to find qwen_tts
    for candidate in [model_dir.parent / "Qwen3-TTS", model_dir.parent]:
        if (candidate / "qwen_tts").is_dir():
            sys.path.insert(0, str(candidate))
            break

    try:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(str(model_dir), fix_mistral_regex=True)
        tokenizer = processor.tokenizer
    except Exception:
        from transformers import Qwen2Tokenizer
        tokenizer = Qwen2Tokenizer.from_pretrained(str(model_dir))

    print(f"Tokenizer vocab size: {len(tokenizer):,}")
    print()

    # ── 1. Check every Malayalam char ───────────────────────────────────────
    ok = 0
    missing = []
    for cp in MALAYALAM_RANGE:
        c = chr(cp)
        try:
            unicodedata.name(c)
        except ValueError:
            continue   # unassigned code point
        ids = tokenizer.encode(c, add_special_tokens=False)
        if len(ids) == 1:
            ok += 1
        else:
            missing.append((c, ids))

    print(f"Single-token Malayalam chars : {ok}")
    if missing:
        print(f"Multi-token fallback chars   : {len(missing)}")
        for c, ids in missing[:5]:
            print(f"  U+{ord(c):04X}  '{c}'  -> {ids}")
    else:
        print("All Malayalam chars tokenise as a single token :)")
    print()

    # ── 2. Sample sentences ─────────────────────────────────────────────────
    print("Sample tokenisations:")
    all_ok = True
    for text in SAMPLE_TEXTS:
        ids   = tokenizer.encode(text, add_special_tokens=False)
        back  = tokenizer.decode(ids)
        match = (back == text)
        if not match:
            all_ok = False
        status = "OK" if match else "MISMATCH"
        print(f"  [{status}]  '{text}'")
        print(f"          tokens={ids}")
        if not match:
            print(f"          decoded='{back}'")

    print()
    if all_ok:
        print("[PASS] Round-trip tokenisation is perfect for all samples.")
    else:
        print("[WARN] Some round-trips had mismatches (check above).")


if __name__ == "__main__":
    main()
