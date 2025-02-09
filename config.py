from datetime import datetime
import locale

locale.getpreferredencoding = lambda: "UTF-8"

# Model configuration
modelname = 'facebook/nllb-200-distilled-1.3B'
source_langs = ["nld", "gos"]
target_langs = source_langs

# Paths
TATOEBA_PATH = 'tatoeba'
modelpath = 'hfacemodels'
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
MODEL_SAVE_PATH = f'/models/nllb-{"-".join(source_langs)}-distilled-1.3B-{timestamp}'

# Training parameters
batch_size = 85
max_chars = 200
max_length = 99
warmup_steps = 100
training_steps = int(1400 * 8 / batch_size)

print('Model save path:', MODEL_SAVE_PATH)