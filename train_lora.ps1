param (
    [string]$QwenDir = "Qwen3-TTS",
    [string]$OutputDir = "output_malayalam_lora",
    [string]$TrainJsonl = "malayalam_data/train_with_codes.jsonl",
    [string]$ValJsonl = "malayalam_data/val_with_codes.jsonl",
    [int]$BatchSize = 1, # Minimal batch size for 4GB VRAM
    [string]$Lr = "2e-6",
    [int]$Epochs = 10,
    [int]$GradAccumSteps = 8, # Increase accumulation to compensate for batch size
    [string]$MixedPrecision = "bf16",
    [string]$AttnImpl = "sdpa", 
    [string]$SpeakerName = "malayalam_speaker",

    [int]$LoraRank = 16,
    [int]$LoraAlpha = 32,
    [float]$LoraDropout = 0.05,
    [string]$LoraBias = "none",
    [string]$LoraTargetModules = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    [string]$InitModelPath = "Qwen3-TTS-Base"
)

$VenvPython = ".venv\Scripts\python.exe"

$ValArgs = @()
if ($ValJsonl -ne "") {
    $ValArgs += "--val_jsonl", $ValJsonl
}

Write-Host "Starting Qwen3-TTS LoRA Fine-Tuning..." -ForegroundColor Cyan

$env:PYTHONPATH = "$QwenDir\finetuning;$env:PYTHONPATH"

& $VenvPython "$QwenDir\finetuning\sft_12hz_lora.py" `
  --init_model_path $InitModelPath `
  --output_model_path $OutputDir `
  --train_jsonl $TrainJsonl `
  --batch_size $BatchSize `
  --lr $Lr `
  --num_epochs $Epochs `
  --speaker_name $SpeakerName `
  --gradient_accumulation_steps $GradAccumSteps `
  --mixed_precision $MixedPrecision `
  --attn_implementation $AttnImpl `
  --lora_rank $LoraRank `
  --lora_alpha $LoraAlpha `
  --lora_dropout $LoraDropout `
  --lora_bias $LoraBias `
  --lora_target_modules $LoraTargetModules `
  @ValArgs
