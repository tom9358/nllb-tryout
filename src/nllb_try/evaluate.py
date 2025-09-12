"""
This script provides functions to evaluate the performance of NLLB models.
It handles translation tasks, calculates standard metrics like BLEU and CHRF scores,
and visualizes these evaluation results across different training steps to
monitor model improvement and potential overfitting.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from sacrebleu import corpus_bleu, corpus_chrf
from .tokenizer_and_model_setup import setup_model_and_tokenizer, cleanup
from .train import preproc  # Import the preprocessing from train script
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


# Calculate metrics for a single DataFrame split
# Calculate metrics for a single DataFrame split
def _calculate_metrics_for_split(df_split: pd.DataFrame, src_lang_nllb: str, tgt_lang_nllb: str, model, tokenizer, sample_size: int = 200):
    if df_split.empty:
        print(f"Warning: The provided DataFrame split for {src_lang_nllb}->{tgt_lang_nllb} is empty. Skipping evaluation for this split. Values set to 0.0.")
        return {
            f"bleu_{src_lang_nllb}_to_{tgt_lang_nllb}": 0.0,
            f"bleu_{tgt_lang_nllb}_to_{src_lang_nllb}": 0.0,
            f"chrf_{src_lang_nllb}_to_{tgt_lang_nllb}": 0.0,
            f"chrf_{tgt_lang_nllb}_to_{src_lang_nllb}": 0.0,
        }

    # Sample data for evaluation (with a sample_size limit for efficiency)
    df_sampled = df_split.sample(n=min(len(df_split), sample_size), random_state=9358)
    src_sentences = df_sampled['source_sentence'].tolist()
    tgt_sentences = df_sampled['target_sentence'].tolist()

    # Translate Source to Target
    translations_src_to_tgt = translate(
        text=src_sentences,
        src_lang=src_lang_nllb,
        tgt_lang=tgt_lang_nllb,
        model=model,
        tokenizer=tokenizer
    )
    bleu_src_to_tgt = corpus_bleu(tgt_sentences, [translations_src_to_tgt]).score
    chrf_src_to_tgt = corpus_chrf(tgt_sentences, [translations_src_to_tgt]).score

    # Translate Target to Source
    translations_tgt_to_src = translate(
        text=preproc(tgt_sentences),
        src_lang=tgt_lang_nllb,
        tgt_lang=src_lang_nllb,
        model=model,
        tokenizer=tokenizer
    )
    bleu_tgt_to_src = corpus_bleu(src_sentences, [translations_tgt_to_src]).score
    chrf_tgt_to_src = corpus_chrf(src_sentences, [translations_tgt_to_src]).score

    return {
        f"bleu_{src_lang_nllb}_to_{tgt_lang_nllb}": bleu_src_to_tgt,
        f"bleu_{tgt_lang_nllb}_to_{src_lang_nllb}": bleu_tgt_to_src,
        f"chrf_{src_lang_nllb}_to_{tgt_lang_nllb}": chrf_src_to_tgt,
        f"chrf_{tgt_lang_nllb}_to_{src_lang_nllb}": chrf_tgt_to_src,
    }

# Main evaluation function calls the helper for each split
def evaluate_model(model, tokenizer, corpus_objects):
    all_corpus_results = []
    for corpus in corpus_objects:
        corpus_results = {}
        
        print(f"  Evaluating {corpus.source_lang_nllb}-{corpus.target_lang_nllb} pair...")

        # Evaluate on Training Set
        train_metrics = _calculate_metrics_for_split(
            corpus.df_train, corpus.source_lang_nllb, corpus.target_lang_nllb, model, tokenizer
        )
        for k, v in train_metrics.items():
            corpus_results[f"train_{k}"] = v

        # Evaluate on Validation Set
        validate_metrics = _calculate_metrics_for_split(
            corpus.df_validate, corpus.source_lang_nllb, corpus.target_lang_nllb, model, tokenizer
        )
        for k, v in validate_metrics.items():
            corpus_results[f"validate_{k}"] = v
            
        all_corpus_results.append(corpus_results)
    return all_corpus_results


def main_evaluate(corpus_objects, MODEL_SAVE_PATH, new_lang_nllb):
    timestamp = config["timestamp"]
    evaldata_folder = 'output/evaluate'
    os.makedirs(evaldata_folder, exist_ok=True)
    
    all_results = {}
    model_versions = [
        d for d in os.listdir(MODEL_SAVE_PATH)
        if os.path.isdir(os.path.join(MODEL_SAVE_PATH, d))
    ]
    model_versions.sort(key=lambda x: int(x))
    for model_name in model_versions:
        print(f"Evaluating model saved at step {model_name}...")
        cleanup()
        model_path = os.path.join(MODEL_SAVE_PATH, model_name)
        model, tokenizer = setup_model_and_tokenizer(model_path, new_lang=new_lang_nllb, device=config['device'])
        
        version_results = evaluate_model(model, tokenizer, corpus_objects)
        
        combined_version_results = {}
        for res_dict in version_results:
            combined_version_results.update(res_dict)
            
        all_results[model_name] = combined_version_results
        
    df_results = pd.DataFrame.from_dict(all_results, orient="index")
    df_results.index.name = "Training Steps"
    df_results.reset_index(inplace=True)
    
    csv_filename = os.path.join(evaldata_folder, f'evaluation_results_{timestamp}.csv')
    df_results.to_csv(csv_filename, index=False)
    
    for corpus in corpus_objects:
        src = corpus.source_lang_nllb
        tgt = corpus.target_lang_nllb

        # Plotting for BLEU Source -> Target
        plot_results(
            df_results,
            f"train_bleu_{src}_to_{tgt}",
            f"validate_bleu_{src}_to_{tgt}",
            f"BLEU Score ({src} \u2192 {tgt})",
            evaldata_folder, timestamp
        )
        # Plotting for BLEU Target -> Source
        plot_results(
            df_results,
            f"train_bleu_{tgt}_to_{src}",
            f"validate_bleu_{tgt}_to_{src}",
            f"BLEU Score ({tgt} \u2192 {src})",
            evaldata_folder, timestamp
        )

        # Plotting for CHRF Source -> Target
        plot_results(
            df_results,
            f"train_chrf_{src}_to_{tgt}",
            f"validate_chrf_{src}_to_{tgt}",
            f"CHRF Score ({src} \u2192 {tgt})",
            evaldata_folder, timestamp
        )
        # Plotting for CHRF Target -> Source
        plot_results(
            df_results,
            f"train_chrf_{tgt}_to_{src}",
            f"validate_chrf_{tgt}_to_{src}",
            f"CHRF Score ({tgt} \u2192 {src})",
            evaldata_folder, timestamp
        )


def plot_results(df_results, metric_train, metric_validate, title, evaldata_folder, timestamp):
    plt.figure(figsize=(12, 6))
    
    if metric_train in df_results.columns:
        plt.plot(df_results["Training Steps"], df_results[metric_train], label="Train", marker='o', linestyle='--')
    if metric_validate in df_results.columns:
        plt.plot(df_results["Training Steps"], df_results[metric_validate], label="Validate", marker='x')

    plt.xlabel("Training Steps")
    plt.ylabel("Score")
    plt.title(f"Model Performance ({title}) by Training Step")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    safe_title = title.replace(' ', '_').replace('(', '').replace(')', '').replace('\u2192', 'to').replace('-', '_')
    plot_filename = os.path.join(evaldata_folder, f"{safe_title.lower()}_plot_{timestamp}.png")
    plt.savefig(plot_filename)
    plt.close()
