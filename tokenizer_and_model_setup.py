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

def setup_model_and_tokenizer(modelname: str, modelpath: str = None, new_lang: str = None, similar_lang: str = None):
    """
    Set up the model and tokenizer for use. This function handles both loading models from HuggingFace and local models,
    adjusts the tokenizer so a new language is supported and optionally sets embedding weights to those of a similar language.
    
    Args:
    - modelname: The model identifier or path.
    - modelpath: The local directory path for caching/loading models.
    - new_lang: Optional; The new language code to add. (In nllb format, e.g. 'gos_Latn')
    - similar_lang: Optional; If provided, the similar language's weights initialize the new language. (nllb format)
    """
    print("torch.cuda.is_available():",torch.cuda.is_available())
    device = torch.device("cuda")# if torch.cuda.is_available() else "cpu")

    # Load the tokenizer
    tokenizer = NllbTokenizer.from_pretrained(modelname, cache_dir=modelpath)
    
    # Fix the tokenizer if a new language is provided
    if new_lang:
        fix_tokenizer(tokenizer, new_lang)
        cleanup()
    else:
        print('Fyi: No new language added')

    # Load the model
    model = AutoModelForSeq2SeqLM.from_pretrained(modelname, cache_dir=modelpath, device_map={"": str(device)})
    
    # Check for the second situation: adding a new language with optional weights initialization
    if new_lang and similar_lang:
        added_token_id = tokenizer.convert_tokens_to_ids(new_lang)
        similar_lang_id = tokenizer.convert_tokens_to_ids(similar_lang)
        # Adjust model weights
        model.model.shared.weight.data[added_token_id] = model.model.shared.weight.data[similar_lang_id]
        print(f'Initialized weights for {new_lang} equal to those of {similar_lang}')
    
    model.resize_token_embeddings(len(tokenizer)) 
    return model, tokenizer