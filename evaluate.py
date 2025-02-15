import os
import pandas as pd
import matplotlib.pyplot as plt
from sacrebleu import corpus_bleu, corpus_chrf
from tokenizer_and_model_setup import setup_model_and_tokenizer, cleanup
from train import preproc  # Import the preprocessing from train script
from config import timestamp

# Ensure the evaldata folder exists
evaldata_folder = 'evaldata'
os.makedirs(evaldata_folder, exist_ok=True)

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

def evaluate_model(model, tokenizer, corpus_objects):
    results = []
    for corpus in corpus_objects:
        df_validate = corpus.df_validate.sample(n=min(len(corpus.df_validate), 200), random_state=9358)
        src_sentences = df_validate['source_sentence'].tolist()
        tgt_sentences = df_validate['target_sentence'].tolist()
        translations_src_to_tgt = translate(
            text=src_sentences,
            src_lang=corpus.source_lang_nllb,
            tgt_lang=corpus.target_lang_nllb,
            model=model,
            tokenizer=tokenizer
        )
        translations_tgt_to_src = translate(
            text=preproc(tgt_sentences),
            src_lang=corpus.target_lang_nllb,
            tgt_lang=corpus.source_lang_nllb,
            model=model,
            tokenizer=tokenizer
        )
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

def main_evaluate(corpus_objects, MODEL_SAVE_PATH, new_lang_nllb):
    all_results = {}
    model_versions = [d for d in os.listdir(MODEL_SAVE_PATH) if os.path.isdir(os.path.join(MODEL_SAVE_PATH, d))]
    for model_name in sorted(model_versions):
        print(f"Evaluating model saved at step {model_name}...")
        model, tokenizer = setup_model_and_tokenizer(MODEL_SAVE_PATH+f"/{model_name}", new_lang_long = new_lang_nllb)
        cleanup()
        version_results = evaluate_model(model, tokenizer, corpus_objects)
        avg_results = pd.DataFrame(version_results).mean().to_dict()
        all_results[model_name] = avg_results
        
    df_results = pd.DataFrame.from_dict(all_results, orient="index")
    df_results.index.name = "Training Steps"
    df_results.reset_index(inplace=True)
    
    csv_filename = os.path.join(evaldata_folder, f'overfitting_investigation_{timestamp}.csv')
    df_results.to_csv(csv_filename)
    
    plot_results(df_results, "bleu_src_to_tgt", "bleu_tgt_to_src", "BLEU Score", timestamp)
    plot_results(df_results, "chrf_src_to_tgt", "chrf_tgt_to_src", "CHRF Score", timestamp)

def plot_results(df_results, metric1, metric2, title, timestamp):
    plt.figure(figsize=(12, 6))
    for metric in [metric1, metric2]:
        plt.plot(df_results["Training Steps"], df_results[metric], label=metric)
    plt.xlabel("Training Steps")
    plt.ylabel("Score")
    plt.title(f"Model Performance ({title}) by Training Step")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_filename = os.path.join(evaldata_folder, f"{title.lower().replace(' ', '_')}_plot_{timestamp}.png")
    plt.savefig(plot_filename)
    plt.close()