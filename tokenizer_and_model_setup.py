from transformers import NllbTokenizer, AutoModelForSeq2SeqLM
from gc import collect
import torch

def fix_tokenizer(tokenizer, new_lang: str):
    """
    Add a new language token to the tokenizer vocabulary.
    This should be done each time after its initialization.
    We need transformers<=4.33 for this to work.
    """
    old_len = len(tokenizer) - int(new_lang in tokenizer.added_tokens_encoder)
    tokenizer.lang_code_to_id[new_lang] = old_len - 1
    tokenizer.id_to_lang_code[old_len - 1] = new_lang
    tokenizer.fairseq_tokens_to_ids["<mask>"] = len(tokenizer.sp_model) + len(tokenizer.lang_code_to_id) + tokenizer.fairseq_offset
    tokenizer.fairseq_tokens_to_ids.update(tokenizer.lang_code_to_id)
    tokenizer.fairseq_ids_to_tokens = {v: k for k, v in tokenizer.fairseq_tokens_to_ids.items()}
    
    if new_lang not in tokenizer._additional_special_tokens:
        tokenizer._additional_special_tokens.append(new_lang)

    tokenizer.added_tokens_encoder = {}
    tokenizer.added_tokens_decoder = {}

def cleanup():
    """Try to free some memory."""
    collect()
    torch.cuda.empty_cache()

def setup_model_and_tokenizer(modelname: str, modelpath: str = None, new_lang_long: str = None, similar_lang_long: str = None):
    """
    Set up the model and tokenizer for use. This function handles both loading models from HuggingFace and local models,
    adjusts the tokenizer so a new language is supported and optionally sets embedding weights to those of a similar language.
    
    Args:
    - modelname: The model identifier or path.
    - modelpath: The local directory path for caching/loading models.
    - new_lang_long: Optional; The new language code to add.
    - similar_lang_long: Optional; If provided, the similar language's weights initialize the new language.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the tokenizer and always fix it if new language is provided
    tokenizer = NllbTokenizer.from_pretrained(modelname, cache_dir=modelpath)
    if new_lang_long is not None:
        fix_tokenizer(tokenizer, new_lang=new_lang_long)
        cleanup()
    
    # Load the model
    model = AutoModelForSeq2SeqLM.from_pretrained(modelname, cache_dir=modelpath, device_map={"": str(device)})
    
    # Only adjust model weights if both new_lang_long and similar_lang_long are provided
    if new_lang_long is not None and similar_lang_long is not None:
        added_token_id = tokenizer.convert_tokens_to_ids(new_lang_long)
        similar_lang_id = tokenizer.convert_tokens_to_ids(similar_lang_long)

        # No need to resize if local models don't need weight adjustment
        model.model.shared.weight.data[added_token_id] = model.model.shared.weight.data[similar_lang_id]
    
    return model, tokenizer