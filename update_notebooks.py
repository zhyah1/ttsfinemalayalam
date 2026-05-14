import json
import os
import re

files_to_check = [
    'inference_hf.ipynb', 
    'train_lora_malayalam.ipynb', 
    'train_lora_malayalam_colab.ipynb', 
    'colab_launcher.ipynb'
]

for file in files_to_check:
    if not os.path.exists(file): 
        continue
        
    with open(file, 'r', encoding='utf-8') as f:
        file_content = f.read()

    # Pre-parse repair for inference_hf.ipynb broken JSON
    if file == 'inference_hf.ipynb' and 'if not os.path.exists(ref_audio):\\n","' in file_content:
        # We replace the completely un-quoted invalid python string representation inside the JSON array.
        file_content = file_content.replace(
            '    if not os.path.exists(ref_audio):\\n","    wavs = glob.glob(\\"**/*.wav\\", recursive=True)\\n",', 
            '                "if not os.path.exists(ref_audio):\\n",\n                "    wavs = glob.glob(\\"**/*.wav\\", recursive=True)\\n",'
        )
    
    try:
        data = json.loads(file_content)
    except json.JSONDecodeError as e:
        print(f"Failed to parse {file}: {e}")
        continue
        
    modified = False
    for cell in data.get('cells', []):
        if cell.get('cell_type') != 'code':
            continue
            
        source_lines = cell.get('source', [])
        source_code = "".join(source_lines)
        original_code = source_code
        
        # 1. Update language="auto" to language="Malayalam"
        source_code = source_code.replace('language="auto"', 'language="Malayalam"')
        source_code = source_code.replace("language='auto'", 'language="Malayalam"')
        
        # 2. Inject language="Malayalam" if missing in generate_voice_clone
        if 'generate_voice_clone(' in source_code and 'language=' not in source_code:
            # Add it right after ref_audio
            source_code = re.sub(
                r'(ref_audio\s*=\s*[^,]+,)', 
                r'\1\n        language="Malayalam",', 
                source_code
            )
            
        # 3. Update the training script call if present
        if '!./train.sh' in source_code:
            replacement = (
                "!python train_malayalam_lora.py \\\n"
                "    --model_dir Qwen3-TTS-Base \\\n"
                "    --train_jsonl malayalam_data/train_with_codes.jsonl \\\n"
                "    --val_jsonl malayalam_data/val_with_codes.jsonl \\\n"
                "    --output_dir output-malayalam-lora \\\n"
                "    --num_epochs 10 \\\n"
                "    --batch_size 1 \\\n"
                "    --lr 2e-6\n"
            )
            source_code = source_code.replace('!./train.sh\n', replacement)
            source_code = source_code.replace('!./train.sh', replacement)
            
        if source_code != original_code:
            # Convert back to list of lines keeping the newlines
            new_lines = [s + '\n' for s in source_code.split('\n')]
            
            # Clean up trailing behaviors
            if not source_code.endswith('\n'):
                new_lines[-1] = new_lines[-1][:-1]
            if new_lines[-1] == '\n' and not source_code.endswith('\n\n'):
                new_lines = new_lines[:-1]
                
            cell['source'] = new_lines
            modified = True
            
    # Write if modified or if we repaired the JSON syntax initially
    with open(file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f'✅ Updated {file}')
