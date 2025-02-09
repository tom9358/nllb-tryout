from transformers import NllbTokenizer, AutoModelForSeq2SeqLM
from gc import collect
import torch

def fix_tokenizer(tokenizer, new_lang='gos_Latn'):
    """
    Add a new language token to the tokenizer vocabulary
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
    """Try to free GPU memory."""
    collect()
    torch.cuda.empty_cache()

def setup_model_and_tokenizer(modelname, modelpath, new_lang_long='gos_Latn', similar_lang_long='nld_Latn'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the tokenizer and fix it
    tokenizer = NllbTokenizer.from_pretrained(modelname, cache_dir=modelpath)
    fix_tokenizer(tokenizer, new_lang=new_lang_long)
    
    # Cleanup to ensure memory is managed properly
    cleanup()
    
    # Load the model
    model = AutoModelForSeq2SeqLM.from_pretrained(modelname, cache_dir=modelpath, device_map={"": str(device)})
    
    # Adjust embeddings for new language token
    added_token_id = tokenizer.convert_tokens_to_ids(new_lang_long)
    similar_lang_id = tokenizer.convert_tokens_to_ids(similar_lang_long)

    # Resizing token embeddings
    model.resize_token_embeddings(len(tokenizer))
    model.model.shared.weight.data[added_token_id + 1] = model.model.shared.weight.data[added_token_id]
    model.model.shared.weight.data[added_token_id] = model.model.shared.weight.data[similar_lang_id]

    return model, tokenizer