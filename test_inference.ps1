param (
    [string]$QwenDir = "Qwen3-TTS",
    [string]$BaseModel = "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    [string]$AdapterDir = "output_lora/checkpoint-epoch-0",
    [string]$Text = "Hello, this is a test of the Qwen3-TTS fine-tuned model.",
    [string]$OutWav = "test_output.wav",
    [float]$LoraScale = 0.3
)

$VenvPython = ".venv\Scripts\python.exe"

Write-Host "Running Inference with LoRA Adapter..." -ForegroundColor Cyan

$env:PYTHONPATH = "$QwenDir\finetuning;$env:PYTHONPATH"

& $VenvPython "$QwenDir\finetuning\infer_lora_custom_voice.py" `

  --base_model_path $BaseModel `
  --adapter_path $AdapterDir `
  --text "$Text" `
  --output_wav $OutWav `
  --lora_scale $LoraScale `
  --attn_implementation "sdpa" # Using sdpa for Windows compatibility
