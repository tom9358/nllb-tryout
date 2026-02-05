from transformers import NllbTokenizer, AutoModelForSeq2SeqLM
from gc import collect
import torch
from .config import config

def cleanup():
    """Try to free some memory."""
    collect()
    torch.cuda.empty_cache()

def setup_model_and_tokenizer(
    modelname: str,
    modelpath: str | None = None,
    new_lang: str | None = None,
    similar_lang: str | None = None,
    device: str = config["device"],
):
    """
    Set up the model and tokenizer for use. This function handles both loading models from HuggingFace and local models,
    adjusts the tokenizer so a new language is supported and optionally sets embedding weights to those of a similar language.

    Args:
    - modelname: The model identifier or path.
    - modelpath: The local directory path for caching/loading models.
    - new_lang: Optional; The new language code to add. (In nllb format, e.g. 'gos_Latn')
    - similar_lang: Optional; If provided, the similar language's weights initialize the new language. (nllb format)
    """
    torch_device = torch.device(device)# if torch.cuda.is_available() else "cpu")

    if new_lang:
        tokenizer = NllbTokenizer.from_pretrained(
            modelname,
            cache_dir=modelpath,
            additional_special_tokens=[new_lang]
        )
    else:
        tokenizer = NllbTokenizer.from_pretrained(modelname, cache_dir=modelpath)

    # Load the model
    model = AutoModelForSeq2SeqLM.from_pretrained(modelname, cache_dir=modelpath, device_map={"": str(torch_device)})
    model.resize_token_embeddings(len(tokenizer))
    # model.tie_weights()

    # Debug
    shared_ptr = model.model.shared.weight.data_ptr()
    enc_ptr = model.model.encoder.embed_tokens.weight.data_ptr()
    dec_ptr = model.model.decoder.embed_tokens.weight.data_ptr()
    lm_ptr = model.lm_head.weight.data_ptr()

    print(f"Tied shared↔encoder: {shared_ptr == enc_ptr} (shared={shared_ptr}, enc={enc_ptr})")
    print(f"Tied shared↔decoder: {shared_ptr == dec_ptr} (shared={shared_ptr}, dec={dec_ptr})")
    print(f"Tied shared↔lm_head: {shared_ptr == lm_ptr} (shared={shared_ptr}, lm={lm_ptr})")
    print("param dtype:", next(model.parameters()).dtype)
    print("param device:", next(model.parameters()).device)

    # Check for the second situation: adding a new language with optional weights initialization
    if new_lang and similar_lang:
        added_token_id = tokenizer.convert_tokens_to_ids(new_lang)
        similar_lang_id = tokenizer.convert_tokens_to_ids(similar_lang)
        # Adjust model weights
        model.model.shared.weight.data[added_token_id] = model.model.shared.weight.data[similar_lang_id]
        print(f'Initialized weights for {new_lang} equal to those of {similar_lang}')

    return model, tokenizer