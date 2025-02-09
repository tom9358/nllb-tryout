import os
from datetime import datetime
import locale
locale.getpreferredencoding = lambda: "UTF-8"

# Model configuration
modelname = 'facebook/nllb-200-distilled-1.3B'
source_langs_tatoeba = ["nld", "gos"]
source_langs_nllb = [lang+'_Latn' for lang in source_langs_tatoeba]
new_lang_nllb = 'gos_Latn'
similar_lang_nllb = 'nld_Latn'

# Paths
TATOEBA_PATH = 'tatoeba'
modelpath = 'hfacemodels'
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
MODEL_SAVE_PATH = f'models/nllb-{"-".join(source_langs_tatoeba)}-distilled-1.3B-{timestamp}'

# Training parameters
batch_size = 65
max_chars = 200
max_length = 99
warmup_steps = 100
training_steps = int(2000 * 8 / batch_size)

print('Model save path:', MODEL_SAVE_PATH)

def save_config_to_file(save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    config_file_path = os.path.join(save_path, 'config.txt')
    with open(config_file_path, 'w', encoding='utf-8') as f:
        f.write(
            f"""
            Model name: {modelname}
            Source languages (Tatoeba): {source_langs_tatoeba}
            Source languages (NLLB): {source_langs_nllb}
            New language (NLLB): {new_lang_nllb}
            Similar language (NLLB): {similar_lang_nllb}
            TATOEBA_PATH: {TATOEBA_PATH}
            Model path: {modelpath}
            Timestamp: {timestamp}
            MODEL_SAVE_PATH: {MODEL_SAVE_PATH}
            Batch size: {batch_size}
            Max chars: {max_chars}
            Max length: {max_length}
            Warmup steps: {warmup_steps}
            Training steps: {training_steps}
            """.strip()
        )

save_config_to_file(MODEL_SAVE_PATH)
