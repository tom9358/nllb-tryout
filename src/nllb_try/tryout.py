import os
from .tokenizer_and_model_setup import setup_model_and_tokenizer, cleanup
from .config import config


def translate(text, src_lang: str, tgt_lang: str, model, tokenizer, a=16, b=1.5, max_input_length: int = 200, **kwargs):
    tokenizer.src_lang = src_lang
    tokenizer.tgt_lang = tgt_lang
    inputs = tokenizer(
        text,
        return_tensors='pt',
        padding='longest',
        truncation=True,
        max_length=max_input_length
    )
    result = model.generate(
        **inputs.to(model.device),
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
        max_new_tokens=int(a + b * inputs.input_ids.shape[1]),
        **kwargs
    )
    return tokenizer.batch_decode(result, skip_special_tokens=True)

def translate_sentences(
    model_save_path: str,
    new_lang_nllb: str,
    sentences_to_translate: list,
    src_lang: str = 'nld_Latn',
    tgt_lang: str = 'gos_Latn'
) -> list:
    # Load the latest model
    model_versions = [
        d for d in os.listdir(model_save_path)
        if os.path.isdir(os.path.join(model_save_path, d))
    ]
    model_versions.sort(key=lambda x: int(x))
    latest_model = model_versions[-1]
    model_path = os.path.join(model_save_path, latest_model)
    print(f"Loading model from {model_path}...")
    model, tokenizer = setup_model_and_tokenizer(model_path, new_lang=new_lang_nllb, device=config['device'])
    print("Model loaded successfully.")

    # List to store translations
    translations = []

    print(f"Translating sentences from {src_lang} to {tgt_lang}...")
    for i, user_input in enumerate(sentences_to_translate):
        if not user_input.strip():
            print(f"Skipping empty sentence at index {i}.")
            translations.append("")
            continue

        try:
            translation = translate(
                text=[user_input],
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                model=model,
                tokenizer=tokenizer
            )[0]
            print(f"Original ({src_lang}): {user_input}")
            print(f"Translation ({tgt_lang}): {translation}\n")
            translations.append(translation)
        except Exception as e:
            print(f"Error during translation for sentence '{user_input}': {e}")
            translations.append(f"ERROR: {e}")
            continue
    cleanup()
    return translations

if __name__ == "__main__":
    main_tryout(config["MODEL_SAVE_PATH"], config["new_lang_nllb"])