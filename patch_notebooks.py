import json
import os

for file in ['inference_hf.ipynb', 'train_lora_malayalam.ipynb', 'train_lora_malayalam_colab.ipynb', 'colab_launcher.ipynb']:
    if not os.path.exists(file): 
        continue
    
    with open(file, 'r', encoding='utf-8') as f: 
        data = json.load(f)
        
    modified = False
    for cell in data.get('cells', []):
        if cell.get('cell_type') != 'code': 
            continue
            
        src = "".join(cell['source'])
        new_src = src
        
        # 1. Update direct git clones
        if '!git clone https://github.com/QwenLM/Qwen3-TTS.git' in new_src and 'qwen3_tts_malayalam.patch' not in new_src:
            new_src = new_src.replace(
                '!git clone https://github.com/QwenLM/Qwen3-TTS.git',
                '!git clone https://github.com/QwenLM/Qwen3-TTS.git\n!cd Qwen3-TTS && git apply ../qwen3_tts_malayalam.patch'
            )
            
        # 2. Update pip install git+ repos
        if "!pip install -q 'git+https://github.com/QwenLM/Qwen3-TTS.git'" in new_src:
            new_src = new_src.replace(
                "!pip install -q 'git+https://github.com/QwenLM/Qwen3-TTS.git'",
                "!git clone https://github.com/QwenLM/Qwen3-TTS.git\n!cd Qwen3-TTS && git apply ../qwen3_tts_malayalam.patch\n!pip install -q -e Qwen3-TTS"
            )
            
        if new_src != src:
            lines = [s + '\n' for s in new_src.split('\n')]
            if not new_src.endswith('\n'): 
                lines[-1] = lines[-1][:-1]
            if lines and lines[-1] == '\n' and not new_src.endswith('\n\n'): 
                lines = lines[:-1]
                
            cell['source'] = lines
            modified = True
            
    if modified:
        with open(file, 'w', encoding='utf-8') as f: 
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f'Patched clone cell in {file}')
