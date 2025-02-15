import os
from datetime import datetime
import locale
locale.getpreferredencoding = lambda: "UTF-8"

# Model configuration
modelname = 'facebook/nllb-200-distilled-600M' #'facebook/nllb-200-distilled-1.3B'
source_langs_tatoeba = ["nld", "gos"]
source_langs_nllb = [lang+'_Latn' for lang in source_langs_tatoeba]
new_lang_nllb = 'gos_Latn'
similar_lang_nllb = 'nld_Latn'

# Paths
TATOEBA_PATH = 'tatoeba'
modelpath = 'hfacemodels'
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
MODEL_SAVE_PATH = f'models/{modelname.split("/")[-1]}-{"-".join(source_langs_tatoeba)}-{timestamp}'

# Training parameters
batch_size = 60
max_chars = 200 # can be set to None
max_length = 99 # tokens
warmup_steps = 100
training_steps = int(2000 * 8 / batch_size)

print('Model save path:', MODEL_SAVE_PATH)

import os

def save_config_to_file(save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    config_params = [
        f"Model name: {modelname}",
        f"Timestamp: {timestamp}",
        f"TATOEBA_PATH: {TATOEBA_PATH}",
        f"MODEL_SAVE_PATH: {MODEL_SAVE_PATH}",
        f"Model path: {modelpath}",
        f"Source languages (Tatoeba): {source_langs_tatoeba}",
        f"Source languages (NLLB): {source_langs_nllb}",
        f"New language (NLLB): {new_lang_nllb}",
        f"Similar language (NLLB): {similar_lang_nllb}",
        f"Batch size: {batch_size}",
        f"Max chars: {max_chars}",
        f"Max length: {max_length}",
        f"Warmup steps: {warmup_steps}",
        f"Training steps: {training_steps}"
    ]

    config_content = "\n".join(config_params)

    config_file_path = os.path.join(save_path, 'config.txt')
    with open(config_file_path, 'w', encoding='utf-8') as f:
        f.write(config_content)

save_config_to_file(MODEL_SAVE_PATH)
