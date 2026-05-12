#!/bin/bash
# Launcher script for Qwen3-TTS Fine-tuning

# Run the data preparation script
echo "Starting data preparation..."
python prepare_malayalam_dataset.py

# Launch training using accelerate
echo "Starting training with accelerate..."
accelerate launch --mixed_precision=bf16 train.py
