import os
import pandas as pd
import matplotlib.pyplot as plt
from transformers import NllbTokenizer, AutoModelForSeq2SeqLM
from sacrebleu import corpus_bleu, corpus_chrf
from tqdm.auto import tqdm
from config import MODEL_SAVE_PATH, timestamp

# Preprocessing setup
def get_non_printing_char_replacer(replace_by: str = " "):
    non_printable_map = {ord(c): replace_by for c in (chr(i) for i in range(sys.maxunicode + 1))
                         if unicodedata.category(c) in {"C", "Cc", "Cf", "Cs", "Co", "Cn"}
                        }
    def replace_non_printing_char(line) -> str:
        return line.translate(non_printable_map)
    return replace_non_printing_char

replace_nonprint = get_non_printing_char_replacer(" ")

mpn = MosesPunctNormalizer(lang="en")
mpn.substitutions = [(re.compile(r), sub) for r, sub in mpn.substitutions]

def preproc(text):
    clean = mpn.normalize(text)
    clean = replace_nonprint(clean)
    clean = unicodedata.normalize("NFKC", clean)
    return clean

def translate(text, src_lang, tgt_lang, model, tokenizer):
    preprocessed_text = [preproc(sentence) for sentence in text]
    tokenizer.src_lang = src_lang
    inputs = tokenizer(preprocessed_text, return_tensors="pt", padding=True, truncation=True).to(model.device)
    outputs = model.generate(**inputs)
    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

def fix_tokenizer(tokenizer, new_langs):
    for new_lang in new_langs:
        old_len = len(tokenizer) - int(new_lang in tokenizer.added_tokens_encoder)
        tokenizer.lang_code_to_id[new_lang] = old_len - 1
        tokenizer.id_to_lang_code[old_len - 1] = new_lang
        tokenizer.fairseq_tokens_to_ids["<mask>"] = len(tokenizer.sp_model) + len(tokenizer.lang_code_to_id) + tokenizer.fairseq_offset
        tokenizer.fairseq_tokens_to_ids.update(tokenizer.lang_code_to_id)
        tokenizer.fairseq_ids_to_tokens = {v: k for k, v in tokenizer.fairseq_tokens_to_ids.items()}
        tokenizer._additional_special_tokens.append(new_lang)
        tokenizer.added_tokens_encoder = {}
        tokenizer.added_tokens_decoder = {}

def load_and_evaluate_model(version_path, corpus_objects):
    model = AutoModelForSeq2SeqLM.from_pretrained(version_path).cuda()
    tokenizer = NllbTokenizer.from_pretrained(version_path)
    fix_tokenizer(tokenizer, [corpus.source_lang_long for corpus in corpus_objects])
    model.resize_token_embeddings(len(tokenizer))
    results = []
    for corpus in corpus_objects:
        df_validate = corpus.df_validate.sample(n=min(len(corpus.df_validate), 200), random_state=9358)
        src_sentences = df_validate['source_sentence'].tolist()
        tgt_sentences = df_validate['target_sentence'].tolist()

        # Translate with preprocessing
        translations_src_to_tgt = translate(
            text=src_sentences,
            src_lang=corpus.source_lang_long,
            tgt_lang=corpus.target_lang_long,
            model=model,
            tokenizer=tokenizer
        )
        translations_tgt_to_src = translate(
            text=tgt_sentences,
            src_lang=corpus.target_lang_long,
            tgt_lang=corpus.source_lang_long,
            model=model,
            tokenizer=tokenizer
        )

        # Calculate BLEU and CHRF metrics
        bleu_src_to_tgt = corpus_bleu(tgt_sentences, [translations_src_to_tgt]).score
        bleu_tgt_to_src = corpus_bleu(src_sentences, [translations_tgt_to_src]).score
        chrf_src_to_tgt = corpus_chrf(tgt_sentences, [translations_src_to_tgt]).score
        chrf_tgt_to_src = corpus_chrf(src_sentences, [translations_tgt_to_src]).score

        results.append({
            "bleu_src_to_tgt": bleu_src_to_tgt,
            "bleu_tgt_to_src": bleu_tgt_to_src,
            "chrf_src_to_tgt": chrf_src_to_tgt,
            "chrf_tgt_to_src": chrf_tgt_to_src,
        })

    return results

def main_evaluate(corpus_objects, model_base_path):
    MODEL_BASE_PATH = 'models/'
    model_versions = [folder_name for folder_name in os.listdir(MODEL_BASE_PATH) if model_base_path[len(MODEL_BASE_PATH):] in folder_name]

    all_results = {}
    for model_name in model_versions:
        step = model_name[-3:]
        print(f"Evaluating model saved at step {step}...")
        version_results = load_and_evaluate_model(MODEL_BASE_PATH+model_name, corpus_objects)
        avg_results = pd.DataFrame(version_results).mean().to_dict()
        all_results[step] = avg_results

    df_results = pd.DataFrame.from_dict(all_results, orient="index")
    df_results.index.name = "Training Steps"
    df_results.reset_index(inplace=True)
    df_results.to_csv(f'overfitting_investigation_{timestamp}.csv')

    plot_results(df_results, "bleu_src_to_tgt", "bleu_tgt_to_src", "BLEU Score")
    plot_results(df_results, "chrf_src_to_tgt", "chrf_tgt_to_src", "CHRF Score")

def plot_results(df_results, metric1, metric2, title):
    plt.figure(figsize=(12, 6))
    for metric in [metric1, metric2]:
        plt.plot(df_results["Training Steps"], df_results[metric], label=metric)
    plt.xlabel("Training Steps")
    plt.ylabel("Score")
    plt.title(f"Model Performance ({title}) by Training Step")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{title.lower().replace(' ', '_')}_plot.png")
    plt.close()