from datetime import datetime
import locale
locale.getpreferredencoding = lambda: "UTF-8"

# Model configuration
modelname = 'facebook/nllb-200-distilled-1.3B'
source_langs = ["afr", "gos"]
target_langs = source_langs

# Paths
TATOEBA_PATH = 'tatoeba'
modelpath = 'hfacemodels'
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
MODEL_SAVE_PATH = f'/models/nllb-{"-".join(source_langs)}-distilled-1.3B-{timestamp}'

print('model save path:', MODEL_SAVE_PATH)