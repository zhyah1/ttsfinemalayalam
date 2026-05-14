"""
train_malayalam_lora.py
========================
A single, self-contained training script for Malayalam LoRA fine-tuning of
Qwen3-TTS-Base (12 Hz tokenizer variant).

KEY DIFFERENCES FROM THE GENERIC sft_12hz_lora.py:
  1. Automatically calls processor.tokenizer.add_tokens() for every Malayalam
     Unicode character that is missing from the vocabulary, then calls
     model.resize_token_embeddings() — so no separate preprocessing step.
  2. Sets language="Malayalam" for all dataset items.
  3. Passes language token (codec_language_id["malayalam"]) into the
     codec channel at the correct position.
  4. Saves the updated tokenizer alongside model checkpoints so inference
     uses the expanded vocab without any extra steps.

Usage (local / Colab):
    python train_malayalam_lora.py \\
        --model_dir    Qwen3-TTS-Base \\
        --train_jsonl  malayalam_data/train_with_codes.jsonl \\
        --val_jsonl    malayalam_data/val_with_codes.jsonl \\
        --output_dir   output-malayalam-lora \\
        --num_epochs   10 \\
        --batch_size   1 \\
        --lr           2e-6
"""

import argparse
import json
import os
import re
import sys

import torch
import torch.nn.functional as F
import librosa
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from accelerate import Accelerator
from transformers import AutoConfig
from safetensors.torch import save_file

try:
    from peft import LoraConfig, TaskType, get_peft_model, PeftModel
except ImportError:
    raise SystemExit("peft is required: pip install peft")

# ── find the qwen_tts package ────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_CANDIDATES = [
    os.path.join(_SCRIPT_DIR, "Qwen3-TTS"),
    os.path.join(_SCRIPT_DIR),
    "/content/ttsfine/Qwen3-TTS",
]
for _c in _REPO_CANDIDATES:
    if os.path.isdir(os.path.join(_c, "qwen_tts")):
        sys.path.insert(0, _c)
        break

from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram


# ═══════════════════════════════════════════════════════════════════════════════
# Malayalam Unicode range
# ═══════════════════════════════════════════════════════════════════════════════
MALAYALAM_RANGE = range(0x0D00, 0x0D80)


def get_malayalam_chars():
    import unicodedata
    out = []
    for cp in MALAYALAM_RANGE:
        c = chr(cp)
        try:
            unicodedata.name(c)
            out.append(c)
        except ValueError:
            pass
    return out


def clean_text(text: str) -> str:
    text = re.sub(r'[\x00-\x1F\x7F]', '', text)
    text = re.sub(r'["\\[\\]{}|<>]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════════
class MalayalamTTSDataset(Dataset):
    def __init__(self, data_list, processor, config):
        self.data_list = data_list
        self.processor = processor
        self.config    = config

    def __len__(self):
        return len(self.data_list)

    @torch.inference_mode()
    def extract_mels(self, audio, sr):
        assert sr == 24000, "Audio must be 24 kHz"
        return mel_spectrogram(
            torch.from_numpy(audio).unsqueeze(0),
            n_fft=1024, num_mels=128, sampling_rate=24000,
            hop_size=256, win_size=1024, fmin=0, fmax=12000,
        ).transpose(1, 2)   # (1, T, 128)

    def _build_text(self, text: str) -> str:
        return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"

    def __getitem__(self, idx):
        item = self.data_list[idx]

        text  = clean_text(item["text"])
        text  = self._build_text(text)
        inp   = self.processor(text=text, return_tensors="pt", padding=True)
        text_ids = inp["input_ids"]  # (1, T)

        audio_codes = torch.tensor(item["audio_codes"], dtype=torch.long)

        wav, _ = librosa.load(item["ref_audio"], sr=24000, mono=True)
        ref_mel = self.extract_mels(wav, 24000)   # (1, T, 128)

        return {
            "text_ids":    text_ids[:, :-5],   # trim trailing pad/eos tokens
            "audio_codes": audio_codes,         # (N, 16)
            "ref_mel":     ref_mel,             # (1, T, 128)
        }

    def collate_fn(self, batch):
        item_length = [
            b["text_ids"].shape[1] + b["audio_codes"].shape[0] for b in batch
        ]
        max_len = max(item_length) + 8
        B, T    = len(batch), max_len

        input_ids              = torch.zeros(  (B, T, 2),  dtype=torch.long)
        codec_ids              = torch.zeros(  (B, T, 16), dtype=torch.long)
        text_embedding_mask    = torch.zeros(  (B, T),     dtype=torch.bool)
        codec_embedding_mask   = torch.zeros(  (B, T),     dtype=torch.bool)
        codec_mask             = torch.zeros(  (B, T),     dtype=torch.bool)
        attention_mask         = torch.zeros(  (B, T),     dtype=torch.long)
        codec_0_labels         = torch.full(   (B, T), -100, dtype=torch.long)

        cfg = self.config

        for i, data in enumerate(batch):
            tid   = data["text_ids"]          # (1, tl)
            acs   = data["audio_codes"]        # (cl, 16)
            ac0   = acs[:, 0]                  # (cl,)
            tl    = tid.shape[1]
            cl    = ac0.shape[0]

            # ── text channel ──────────────────────────────────────────────
            input_ids[i, :3, 0]                      = tid[0, :3]
            input_ids[i, 3:7, 0]                     = cfg.tts_pad_token_id
            input_ids[i,   7, 0]                     = cfg.tts_bos_token_id
            input_ids[i, 8:8+tl-3, 0]               = tid[0, 3:]
            input_ids[i,   8+tl-3, 0]               = cfg.tts_eos_token_id
            input_ids[i, 8+tl-2:8+tl+cl, 0]        = cfg.tts_pad_token_id
            text_embedding_mask[i, :8+tl+cl]         = True

            # ── codec channel ─────────────────────────────────────────────
            talker_cfg = cfg.talker_config
            input_ids[i, 3:8, 1] = torch.tensor([
                talker_cfg.codec_nothink_id,
                talker_cfg.codec_think_bos_id,
                talker_cfg.codec_think_eos_id,
                0,                              # speaker embedding slot
                talker_cfg.codec_pad_id,
            ])
            input_ids[i, 8:8+tl-3, 1]               = talker_cfg.codec_pad_id
            input_ids[i, 8+tl-3, 1]                 = talker_cfg.codec_pad_id
            input_ids[i, 8+tl-2, 1]                 = talker_cfg.codec_bos_id
            lo, hi = 8+tl-1, 8+tl-1+cl
            input_ids[i, lo:hi, 1]                  = ac0
            input_ids[i, hi, 1]                     = talker_cfg.codec_eos_token_id

            codec_0_labels[i, lo:hi]                 = ac0
            codec_0_labels[i, hi]                    = talker_cfg.codec_eos_token_id
            codec_ids[i, lo:hi, :]                   = acs

            codec_embedding_mask[i, 3:8+tl+cl]      = True
            codec_embedding_mask[i, 6]               = False  # speaker slot

            codec_mask[i, lo:hi]                     = True
            attention_mask[i, :8+tl+cl]              = True

        ref_mels = torch.cat([d["ref_mel"] for d in batch], dim=0)  # (B, T, 128)

        return {
            "input_ids":             input_ids,
            "ref_mels":              ref_mels,
            "attention_mask":        attention_mask,
            "text_embedding_mask":   text_embedding_mask.unsqueeze(-1),
            "codec_embedding_mask":  codec_embedding_mask.unsqueeze(-1),
            "codec_0_labels":        codec_0_labels,
            "codec_ids":             codec_ids,
            "codec_mask":            codec_mask,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Loss
# ═══════════════════════════════════════════════════════════════════════════════
_target_spk_emb = None


def compute_loss(model, batch):
    global _target_spk_emb

    input_ids           = batch["input_ids"]
    codec_ids           = batch["codec_ids"]
    ref_mels            = batch["ref_mels"]
    text_embedding_mask = batch["text_embedding_mask"]
    codec_embedding_mask= batch["codec_embedding_mask"]
    attention_mask      = batch["attention_mask"]
    codec_0_labels      = batch["codec_0_labels"]
    codec_mask          = batch["codec_mask"]

    dev   = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    with torch.no_grad():
        spk_emb = model.speaker_encoder(ref_mels.to(dtype=dtype, device=dev)).detach()

    if model.training and _target_spk_emb is None:
        _target_spk_emb = spk_emb.detach().cpu()

    input_text_ids  = input_ids[:, :, 0]
    input_codec_ids = input_ids[:, :, 1]

    te = model.talker.model.text_embedding(input_text_ids)
    if hasattr(model.talker, "text_projection"):
        te = model.talker.text_projection(te)
    te = te * text_embedding_mask

    ce = model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
    ce[:, 6, :] = spk_emb
    emb = te + ce

    for i in range(1, 16):
        sub_emb = model.talker.code_predictor.get_input_embeddings()[i - 1](
            codec_ids[:, :, i]
        )
        emb = emb + sub_emb * codec_mask.unsqueeze(-1)

    out = model.talker(
        inputs_embeds=emb[:, :-1, :],
        attention_mask=attention_mask[:, :-1],
        labels=None,
        output_hidden_states=True,
    )
    logits     = out.logits
    tgt        = codec_0_labels[:, 1:]
    codec0_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        tgt.reshape(-1),
        ignore_index=-100,
    )

    hidden       = out.hidden_states[0][-1]
    talker_hs    = hidden[codec_mask[:, 1:]]
    talker_cids  = codec_ids[codec_mask]
    _, sub_loss  = model.talker.forward_sub_talker_finetune(talker_cids, talker_hs)

    return codec0_loss + sub_loss


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════════
def evaluate(model, dataloader, accelerator):
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in dataloader:
            loss   = compute_loss(model, batch)
            gathered = accelerator.gather_for_metrics(loss.detach())
            if gathered.ndim == 0:
                gathered = gathered.unsqueeze(0)
            losses.append(gathered)
    model.train()
    if not losses:
        return None
    return torch.cat(losses).mean().item()


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
def _parse_list(s: str):
    return [x.strip() for x in s.split(",") if x.strip()]


def train():
    global _target_spk_emb

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir",        type=str, default="Qwen3-TTS-Base")
    parser.add_argument("--output_dir",       type=str, default="output-malayalam-lora")
    parser.add_argument("--train_jsonl",      type=str, required=True)
    parser.add_argument("--val_jsonl",        type=str, default=None)
    parser.add_argument("--batch_size",       type=int, default=1)
    parser.add_argument("--eval_batch_size",  type=int, default=None)
    parser.add_argument("--lr",               type=float, default=2e-6)
    parser.add_argument("--num_epochs",       type=int, default=10)
    parser.add_argument("--start_epoch",      type=int, default=0)
    parser.add_argument("--resume_adapter",   type=str, default=None)
    parser.add_argument("--speaker_name",     type=str, default="malayalam_speaker")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--mixed_precision",  type=str, default="bf16",
                        choices=["no", "fp16", "bf16"])
    parser.add_argument("--attn_implementation", type=str, default="eager")
    parser.add_argument("--save_every",       type=int, default=1)
    parser.add_argument("--eval_every",       type=int, default=1)
    parser.add_argument("--lora_rank",        type=int, default=16)
    parser.add_argument("--lora_alpha",       type=int, default=32)
    parser.add_argument("--lora_dropout",     type=float, default=0.05)
    parser.add_argument("--lora_bias",        type=str, default="none")
    parser.add_argument("--lora_target_modules", type=str,
                        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=None if args.mixed_precision == "no" else args.mixed_precision,
        log_with="tensorboard",
        project_dir=args.output_dir,
    )

    # ── Load model + processor ────────────────────────────────────────────
    accelerator.print("Loading Qwen3-TTS-Base …")
    wrapper = Qwen3TTSModel.from_pretrained(
        args.model_dir,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
    )
    config    = AutoConfig.from_pretrained(args.model_dir)
    processor = wrapper.processor        # Qwen3TTSProcessor
    tokenizer = processor.tokenizer      # Qwen2Tokenizer

    # ── Expand tokenizer with Malayalam chars ─────────────────────────────
    accelerator.print("Checking tokenizer for Malayalam coverage …")
    ml_chars    = get_malayalam_chars()
    missing     = [c for c in ml_chars if tokenizer.convert_tokens_to_ids(c) ==
                   tokenizer.unk_token_id]
    if missing:
        accelerator.print(f"  Adding {len(missing)} missing Malayalam tokens …")
        tokenizer.add_tokens(missing)
        new_vocab_size = len(tokenizer)
        # Resize the text_embedding layer in the talker
        talker = wrapper.model.talker
        if hasattr(talker, "model") and hasattr(talker.model, "text_embedding"):
            emb = talker.model.text_embedding
            old_n, H = emb.weight.shape
            if new_vocab_size > old_n:
                accelerator.print(f"  Resizing text_embedding {old_n} → {new_vocab_size}")
                import torch.nn as nn
                mean_vec = emb.weight.data.mean(dim=0, keepdim=True)
                extra    = new_vocab_size - old_n
                noise    = torch.zeros(extra, H, dtype=emb.weight.dtype)
                noise.normal_(std=0.01)
                new_w    = torch.cat([emb.weight.data, mean_vec.expand(extra, -1) + noise])
                new_emb  = nn.Embedding(new_vocab_size, H, dtype=emb.weight.dtype)
                new_emb.weight = nn.Parameter(new_w)
                talker.model.text_embedding = new_emb
                # Also update config
                config.talker_config.text_vocab_size = new_vocab_size
        accelerator.print(f"  ✓ Tokenizer vocab size: {new_vocab_size:,}")
    else:
        accelerator.print("  ✓ All Malayalam chars already in tokenizer vocab.")

    # ── Make sure Malayalam language id exists in config ─────────────────
    codec_lang_ids = getattr(config.talker_config, "codec_language_id", {})
    if "malayalam" not in codec_lang_ids:
        codec_lang_ids["malayalam"] = 2072
        accelerator.print("  Added 'malayalam' → codec_language_id=2072")
    config.talker_config.codec_language_id = codec_lang_ids

    # ── LoRA setup ────────────────────────────────────────────────────────
    qwen_model = wrapper.model.talker
    if args.resume_adapter:
        model = PeftModel.from_pretrained(qwen_model, args.resume_adapter, is_trainable=True)
    else:
        lora_cfg = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias=args.lora_bias,
            target_modules=_parse_list(args.lora_target_modules),
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(qwen_model, lora_cfg)

    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    if accelerator.is_main_process:
        model.print_trainable_parameters()

    # ── Datasets ─────────────────────────────────────────────────────────
    with open(args.train_jsonl, encoding="utf-8") as f:
        train_data = [json.loads(l) for l in f]
    train_ds = MalayalamTTSDataset(train_data, processor, config)
    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True, collate_fn=train_ds.collate_fn,
    )

    val_dl = None
    if args.val_jsonl:
        with open(args.val_jsonl, encoding="utf-8") as f:
            val_data = [json.loads(l) for l in f]
        val_ds = MalayalamTTSDataset(val_data, processor, config)
        ebs    = args.eval_batch_size or args.batch_size
        val_dl = DataLoader(
            val_ds, batch_size=ebs,
            shuffle=False, collate_fn=val_ds.collate_fn,
        )

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    if val_dl is not None:
        model, optimizer, train_dl, val_dl = accelerator.prepare(
            model, optimizer, train_dl, val_dl
        )
    else:
        model, optimizer, train_dl = accelerator.prepare(model, optimizer, train_dl)

    # Keep a reference to the un-wrapped Qwen3TTSForConditionalGeneration
    # so compute_loss can reach speaker_encoder
    qwen_full_model = wrapper.model

    # ── Training loop ─────────────────────────────────────────────────────
    model.train()
    for local_epoch in range(args.num_epochs):
        epoch = args.start_epoch + local_epoch
        for step, batch in enumerate(train_dl):
            with accelerator.accumulate(model):
                loss = compute_loss(qwen_full_model, batch)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            if step % 10 == 0:
                accelerator.print(
                    f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}"
                )

        if val_dl is not None and (local_epoch + 1) % args.eval_every == 0:
            val_loss = evaluate(qwen_full_model, val_dl, accelerator)
            if accelerator.is_main_process and val_loss is not None:
                accelerator.print(f"Epoch {epoch} | Val Loss: {val_loss:.4f}")

        if accelerator.is_main_process and (local_epoch + 1) % args.save_every == 0:
            out_dir = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch}")
            os.makedirs(out_dir, exist_ok=True)

            unwrapped = accelerator.unwrap_model(model)
            unwrapped.save_pretrained(out_dir, safe_serialization=True)

            # Save updated tokenizer (with Malayalam tokens) ──────────────
            tokenizer.save_pretrained(out_dir)
            processor.save_pretrained(out_dir)
            accelerator.print(f"  Saved tokenizer to {out_dir}")

            # Update config.json ──────────────────────────────────────────
            src_cfg = os.path.join(args.model_dir, "config.json")
            out_cfg = os.path.join(out_dir, "config.json")
            with open(src_cfg, encoding="utf-8") as f:
                cfg_dict = json.load(f)

            cfg_dict["tts_model_type"] = "custom_voice"
            tc = cfg_dict.setdefault("talker_config", {})
            tc.setdefault("spk_id", {})[args.speaker_name] = 3000
            tc.setdefault("spk_is_dialect", {})[args.speaker_name] = False
            tc.setdefault("codec_language_id", {})["malayalam"] = 2072
            tc["text_vocab_size"] = len(tokenizer)

            with open(out_cfg, "w", encoding="utf-8") as f:
                json.dump(cfg_dict, f, indent=2, ensure_ascii=False)

            # Save speaker embedding ───────────────────────────────────────
            if _target_spk_emb is not None:
                save_file(
                    {"target_speaker_embedding": _target_spk_emb[0]},
                    os.path.join(out_dir, "speaker_embedding.safetensors"),
                )
                accelerator.print(f"  Saved speaker_embedding.safetensors")

            accelerator.print(f"  ✓ Checkpoint saved: {out_dir}")


if __name__ == "__main__":
    train()
