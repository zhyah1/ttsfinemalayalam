import json
import os

for file in ['train_lora_malayalam.ipynb', 'train_lora_malayalam_colab.ipynb', 'colab_launcher.ipynb']:
    if not os.path.exists(file): 
        continue
    
    with open(file, 'r', encoding='utf-8') as f: 
        data = json.load(f)
    
    modified = False
    for cell in data.get('cells', []):
        if cell.get('cell_type') != 'code': 
            continue
            
        src = ''.join(cell['source'])
        
        # if the training command is present, but the prepare script is NOT present
        if '!python train_malayalam_lora.py' in src and 'prepare_malayalam_dataset.py' not in src:
            prepare_cmd = (
                "!python prepare_malayalam_dataset.py \\\n"
                "    --dataset_name siyah1/Malayalam-TTS-v2 \\\n"
                "    --output_dir malayalam_data \\\n"
                "    --tokenizer_path Qwen3-TTS-Base\n\n"
            )
            src = src.replace('!python train_malayalam_lora.py', prepare_cmd + '!python train_malayalam_lora.py')
            
            lines = [s + '\n' for s in src.split('\n')]
            if not src.endswith('\n'): 
                lines[-1] = lines[-1][:-1]
            if lines and lines[-1] == '\n' and not src.endswith('\n\n'): 
                lines = lines[:-1]
            
            cell['source'] = lines
            modified = True
            
    if modified:
        with open(file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f'Added dataset preparation to {file}')
