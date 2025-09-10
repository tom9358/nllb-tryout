import os
from datetime import datetime
import locale
locale.getpreferredencoding = lambda: "UTF-8"

# Default Configuration
config = {
    # Model configuration
    'modelname': 'facebook/nllb-200-distilled-1.3B',  # or e.g. 'facebook/nllb-200-distilled-600M'
    'source_langs_tatoeba': ["nld", "gos"],
    'source_langs_nllb': ["nld_Latn", "gos_Latn"],
    'new_lang_nllb': 'gos_Latn',
    'similar_lang_nllb': 'nld_Latn',
    
    # Paths
    'DATA_ROOT_PATH': 'data', # Root for all data
    'TATOEBA_PATH': os.path.join('data', 'tatoeba'), # Relative to DATA_ROOT_PATH
    'modelpath': 'hfacemodels',
    'timestamp': datetime.now().strftime("%Y%m%d-%H%M%S"),
    
    # Training parameters
    'batch_size': 34,
    'max_chars': 200,       # Can be set to None
    'max_length': 100,       # tokens
    'warmup_steps': 100,
    'training_steps': 1000,
    'device': 'cuda',
}
config['MODEL_SAVE_PATH'] = f'checkpoints/{config["modelname"].split("/")[-1]}-{"-".join(config["source_langs_tatoeba"])}-{config["timestamp"]}'

def save_config_to_file(save_path):
    """
    Save the current configuration to a config.txt file.
    
    Args:
        save_path (str): The directory where the config file will be saved.
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    config_params = [
        f"Model name: {config['modelname']}",
        f"Timestamp: {config['timestamp']}",
        f"TATOEBA_PATH: {config['TATOEBA_PATH']}",
        f"MODEL_SAVE_PATH: {config['MODEL_SAVE_PATH']}",
        f"Model path: {config['modelpath']}",
        f"Source languages (Tatoeba): {config['source_langs_tatoeba']}",
        f"Source languages (NLLB): {config['source_langs_nllb']}",
        f"New language (NLLB): {config['new_lang_nllb']}",
        f"Similar language (NLLB): {config['similar_lang_nllb']}",
        f"Batch size: {config['batch_size']}",
        f"Max chars: {config['max_chars']}",
        f"Max length: {config['max_length']}",
        f"Warmup steps: {config['warmup_steps']}",
        f"Training steps: {config['training_steps']}"
    ]

    config_content = "\n".join(config_params)

    config_file_path = os.path.join(save_path, 'config.txt')
    with open(config_file_path, 'w', encoding='utf-8') as f:
        f.write(config_content)

    print('Configuration saved to:', config_file_path)
    print('Model save path:', config['MODEL_SAVE_PATH'])

save_config_to_file(config['MODEL_SAVE_PATH'])
