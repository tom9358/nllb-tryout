from datetime import datetime
import locale
locale.getpreferredencoding = lambda: "UTF-8"

# Model configuration
modelpath = 'hfacemodels'
modelname = 'facebook/nllb-200-distilled-1.3B'

# Language codes
source_langs = ["nld", "gos"]
target_langs = source_langs

# Paths
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
MODEL_SAVE_PATH = f'/models/nllb-{"-".join(source_langs)}-distilled-1.3B-{timestamp}'
print('model save path:', MODEL_SAVE_PATH)