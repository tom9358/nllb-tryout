import os
from tokenizer_and_model_setup import setup_model_and_tokenizer, cleanup
from config import MODEL_SAVE_PATH, new_lang_nllb

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

def main_tryout(model_save_path: str, new_lang_nllb: str, inputlist: list = None):
    # Load the latest model
    model_versions = [
        d for d in os.listdir(model_save_path)
        if os.path.isdir(os.path.join(model_save_path, d))
    ]
    model_versions.sort(key=lambda x: int(x))
    latest_model = model_versions[-1]
    model_path = os.path.join(model_save_path, latest_model)
    print(f"Loading model from {model_path}...")
    model, tokenizer = setup_model_and_tokenizer(model_path, new_lang=new_lang_nllb)
    print("Model loaded successfully.")

    # Set default languages
    src_lang = 'nld_Latn'
    tgt_lang = 'gos_Latn'
    print("Enter text to translate. Type 'END' to exit or use 'LANG src tgt' to change languages. Format e.g. nld_Latn gos_Latn")

    input_iterator = iter(inputlist) if inputlist else None

    while True:
        if input_iterator:
            try:
                user_input = next(input_iterator)
                print(f"Translate ({src_lang} -> {tgt_lang}): {user_input}")
            except StopIteration:
                input_iterator = None  # Switch to interactive mode after list is exhausted
                continue
        else:
            user_input = input(f"Translate ({src_lang} -> {tgt_lang}): ")

        if user_input == "END":
            print("Exiting translation tool.")
            break
        elif user_input.startswith("LANG"):
            parts = user_input.split()
            if len(parts) == 3:
                _, new_src, new_tgt = parts
                src_lang = new_src
                tgt_lang = new_tgt
                print(f"Language pair updated to: {src_lang} -> {tgt_lang}")
            else:
                print("Invalid LANG command. Use 'LANG src tgt'.")
            continue
        elif user_input == "":
            continue  # Skip empty input

        try:
            translation = translate(
                text=[user_input],
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                model=model,
                tokenizer=tokenizer
            )[0]
            print(f"Translation: {translation}")
        except Exception as e:
            print(f"Error during translation: {e}")

    cleanup()

if __name__ == "__main__":
    main_tryout(MODEL_SAVE_PATH, new_lang_nllb)