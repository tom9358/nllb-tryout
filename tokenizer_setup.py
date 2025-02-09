from transformers import NllbTokenizer, AutoModelForSeq2SeqLM

def fix_tokenizer(tokenizer, new_lang='gos_Latn'):
    old_len = len(tokenizer) - int(new_lang in tokenizer.added_tokens_encoder)
    tokenizer.lang_code_to_id[new_lang] = old_len - 1
    tokenizer.id_to_lang_code[old_len - 1] = new_lang
    tokenizer.fairseq_tokens_to_ids["<mask>"] = len(tokenizer.sp_model) + len(tokenizer.lang_code_to_id) + tokenizer.fairseq_offset
    tokenizer.fairseq_tokens_to_ids.update(tokenizer.lang_code_to_id)
    tokenizer.fairseq_ids_to_tokens = {v: k for k, v in tokenizer.fairseq_tokens_to_ids.items()}
    tokenizer._additional_special_tokens.append(new_lang)
    tokenizer.added_tokens_encoder = {}
    tokenizer.added_tokens_decoder = {}

def setup_model_and_tokenizer(modelname, modelpath):
    tokenizer = NllbTokenizer.from_pretrained(modelname, cache_dir=modelpath)
    fix_tokenizer(tokenizer)
    model = AutoModelForSeq2SeqLM.from_pretrained(modelname, cache_dir=modelpath, device_map={"": "cuda:0"})
    return model, tokenizer