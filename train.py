import os, sys, json, torch, librosa, numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import torch.nn.functional as F
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoConfig
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram

# -- Config --
MODEL_DIR = '/content/ttsfine/Qwen3-TTS-Base'
TRAIN_JSON = '/content/ttsfine/malayalam_data/train_with_codes.jsonl'
OUTPUT_DIR = '/content/output_malayalam_lora'
os.makedirs(OUTPUT_DIR, exist_ok=True)

class TTSDataset(Dataset):
    def __init__(self, data_list, processor, config):
        self.data_list = data_list
        self.processor = processor
        self.config = config

    def __len__(self): return len(self.data_list)

    @torch.inference_mode()
    def extract_mels(self, audio, sr):
        return mel_spectrogram(torch.from_numpy(audio).unsqueeze(0), n_fft=1024, num_mels=128, sampling_rate=24000, hop_size=256, win_size=1024, fmin=0, fmax=12000).transpose(1, 2)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        text = f"<|im_start|>assistant\n{item['text']}<|im_end|>\n<|im_start|>assistant\n"
        text_ids = self.processor(text=text, return_tensors="pt")["input_ids"]
        audio_codes = torch.tensor(item["audio_codes"], dtype=torch.long)
        wav, _ = librosa.load(item["ref_audio"], sr=24000)
        ref_mel = self.extract_mels(wav, 24000)
        return {"text_ids": text_ids[0][:-5], "audio_codes": audio_codes, "ref_mel": ref_mel}

    def collate_fn(self, batch):
        item_length = [b["text_ids"].shape[0] + b["audio_codes"].shape[0] for b in batch]
        max_length = max(item_length) + 16
        b, t = len(batch), max_length
        input_ids = torch.zeros((b,t,2), dtype=torch.long)
        codec_ids = torch.zeros((b,t,16), dtype=torch.long)
        text_embedding_mask = torch.zeros((b,t), dtype=torch.bool)
        codec_embedding_mask = torch.zeros((b,t), dtype=torch.bool)
        codec_mask = torch.zeros((b,t), dtype=torch.bool)
        attention_mask = torch.zeros((b,t), dtype=torch.long)
        codec_0_labels = torch.full((b,t), -100, dtype=torch.long)
        
        for i, data in enumerate(batch):
            tid, acs = data["text_ids"], data["audio_codes"]
            ac0 = acs[:,0]
            tl, cl = tid.shape[0], ac0.shape[0]
            input_ids[i,:tl,0] = tid
            input_ids[i,tl:tl+cl,0] = self.config.tts_pad_token_id
            text_embedding_mask[i,:tl+cl] = True
            input_ids[i,tl,1] = self.config.talker_config.codec_bos_id
            input_ids[i,tl:tl+cl,1] = ac0
            codec_0_labels[i,tl:tl+cl] = ac0
            codec_ids[i,tl:tl+cl,:] = acs
            codec_embedding_mask[i,tl:tl+cl] = True
            codec_mask[i,tl:tl+cl] = True
            attention_mask[i,:tl+cl] = True
            
        ref_mels = torch.cat([d["ref_mel"] for d in batch], dim=0)
        return {"input_ids":input_ids, "ref_mels":ref_mels, "attention_mask":attention_mask,
                "text_embedding_mask":text_embedding_mask.unsqueeze(-1), "codec_embedding_mask":codec_embedding_mask.unsqueeze(-1),
                "codec_0_labels":codec_0_labels, "codec_ids":codec_ids, "codec_mask":codec_mask}

def compute_loss(qwen_wrapper, batch):
    model = qwen_wrapper.model
    input_ids = batch['input_ids'].to(model.device)
    codec_0_labels = batch['codec_0_labels'].to(model.device)
    ref_mels  = batch['ref_mels'].to(model.device)
    text_embedding_mask = batch['text_embedding_mask'].to(model.device)
    codec_embedding_mask = batch['codec_embedding_mask'].to(model.device)
    attention_mask = batch['attention_mask'].to(model.device)

    with torch.no_grad():
        # Correct path for speaker_encoder is in the main model
        if hasattr(qwen_wrapper.model, 'speaker_encoder'):
            encoder = qwen_wrapper.model.speaker_encoder
        elif hasattr(qwen_wrapper, 'speaker_encoder'):
            encoder = qwen_wrapper.speaker_encoder
        else:
            raise AttributeError("Could not find speaker_encoder in model or wrapper.")
            
        spk_emb = encoder(ref_mels.to(dtype=next(encoder.parameters()).dtype)).detach()

    te = model.talker.model.text_embedding(input_ids[:,:,0])
    if hasattr(model.talker, 'text_projection'): te = model.talker.text_projection(te)
    te = te * text_embedding_mask
    ce = model.talker.model.codec_embedding(input_ids[:,:,1]) * codec_embedding_mask
    ce[:, 6, :] = spk_emb
    emb = te + ce

    out = model.talker(inputs_embeds=emb[:, :-1, :], attention_mask=attention_mask[:, :-1])
    loss = F.cross_entropy(out.logits.reshape(-1, out.logits.size(-1)), codec_0_labels[:, 1:].reshape(-1), ignore_index=-100)
    return loss

def main():
    accelerator = Accelerator(gradient_accumulation_steps=8, mixed_precision='bf16')
    
    qwen_base = Qwen3TTSModel.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16)
    
    lora_config = LoraConfig(
        r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], 
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(qwen_base.model, lora_config)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})

    with open(TRAIN_JSON, 'r') as f: data = [json.loads(line) for line in f]
    dataset = TTSDataset(data, qwen_base.processor, qwen_base.model.config)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=dataset.collate_fn)

    optimizer = AdamW(model.parameters(), lr=2e-6)
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)

    for epoch in range(10):
        model.train()
        for step, batch in enumerate(loader):
            with accelerator.accumulate(model):
                loss = compute_loss(qwen_base, batch)
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
            if step % 10 == 0:
                accelerator.print(f"Epoch {epoch} | Step {step} | Loss {loss.item()}")
        
        if accelerator.is_main_process:
            model.save_pretrained(os.path.join(OUTPUT_DIR, f"checkpoint-epoch-{epoch}"))

if __name__ == "__main__":
    main()
