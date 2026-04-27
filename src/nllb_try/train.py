import torch
from transformers import Adafactor, get_constant_schedule_with_warmup
import numpy as np
from .tokenizer_and_model_setup import setup_model_and_tokenizer, cleanup
from .config import RunConfig, get_default_config
from .seed import set_seed
from .artifacts import (
    format_run_config_txt,
    init_run_dir,
    write_json,
    write_loss_csv,
)
from .augmentation import (
    preproc,
    apply_variations,
    add_gronings_variations,
    swap_synonyms,
    synonym_pairs_gos,
)
from tqdm.auto import trange
import pandas as pd
import matplotlib.pyplot as plt


def get_balanced_df(
    corpus_objects: list,
    temperature: float = 5.0,
    target_total_samples: int | None = None,
    verbose: bool = False,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Sample from each corpus using temperature-based balancing.

    With *T=1* you get proportional sampling (status quo).
    With *T → ∞* every corpus contributes equally.
    *T=5* is a standard middle-ground for multilingual MT (NLLB / M2M-100).

    Returns (df, src_langs, tgt_langs) where *df* has columns
    ``source_sentence``, ``target_sentence``, and ``corpus_idx``.
    """
    counts = np.array([len(c.df_train) for c in corpus_objects], dtype=float)
    probs = counts ** (1.0 / temperature)
    probs /= probs.sum()

    if verbose:
        print(f"Balanced sampling with temperature={temperature}")

    if target_total_samples is None:
        target_total_samples = int(counts.sum())

    dfs: list[pd.DataFrame] = []
    src_langs_all: list[str] = []
    tgt_langs_all: list[str] = []

    for i, corpus in enumerate(corpus_objects):
        n_samples = int(probs[i] * target_total_samples)
        n_samples = max(n_samples, 1)  # always at least one sample

        replace = n_samples > len(corpus.df_train)
        sampled = corpus.df_train.sample(n=n_samples, replace=replace)
        sampled = sampled.copy()
        sampled["corpus_idx"] = i
        dfs.append(sampled)
        src_langs_all.extend([corpus.source_lang_nllb] * n_samples)
        tgt_langs_all.extend([corpus.target_lang_nllb] * n_samples)

        if verbose:
            pct = probs[i] * 100
            ratio = n_samples / len(corpus.df_train)
            direction = "oversampled" if ratio > 1 else "undersampled" if ratio < 1 else "exact"
            print(
                f"  Corpus {i} ({corpus.source_lang_nllb}→{corpus.target_lang_nllb}): "
                f"{len(corpus.df_train):,} original → {n_samples:,} sampled "
                f"({pct:.1f}%, {ratio:.2f}x, {direction})"
            )

    df = pd.concat(dfs).reset_index(drop=True)
    return df, np.array(src_langs_all, dtype=object), np.array(tgt_langs_all, dtype=object)


def tokenize_mixed_langs(
    tokenizer, texts: list[str], langs: list[str], max_length: int, device
) -> tuple[torch.Tensor, torch.Tensor]:
    # Returns (input_ids, attention_mask) stacked tensors (len(texts), max_length)
    # Tokenizes a list of sentences and corresponding languages efficiently, handling mixed language batches by grouping sentences by language for faster batched tokenization.
    # Required for multilingual datasets when tokenizer must know the language per sentence (which NLLB does), allowing bulk tokenization while respecting per-sentence language settings.
    idxs_by_lang: dict[str, list[int]] = {}
    for i, lang in enumerate(langs):
        idxs_by_lang.setdefault(lang, []).append(i)
    input_ids_dict: dict[int, torch.Tensor] = {}
    attention_mask_dict: dict[int, torch.Tensor] = {}
    for lang, idxs in idxs_by_lang.items():
        batch_texts = [texts[i] for i in idxs]
        tokenizer.src_lang = lang
        feats = tokenizer(
            batch_texts,
            return_tensors='pt',
            truncation=True,
            padding='max_length',
            max_length=max_length
        )
        for j, i_global in enumerate(idxs):
            input_ids_dict[i_global] = feats['input_ids'][j]
            attention_mask_dict[i_global] = feats['attention_mask'][j]
    input_ids: list[torch.Tensor] = [input_ids_dict[i] for i in range(len(texts))]
    attention_mask: list[torch.Tensor] = [attention_mask_dict[i] for i in range(len(texts))]
    input_ids_tensor = torch.stack(input_ids).to(device)
    attention_mask_tensor = torch.stack(attention_mask).to(device)
    return input_ids_tensor, attention_mask_tensor

def train_model(model, tokenizer, corpus_objects: list, cfg: RunConfig) -> None:
    batch_size: int = cfg.batch_size
    max_length: int = cfg.max_length
    num_epochs: int = cfg.num_epochs
    warmup_steps: int = cfg.warmup_steps

    paths = init_run_dir(cfg.run_dir)
    checkpoints_dir = paths["checkpoints_dir"]
    train_dir = paths["train_dir"]

    device = next(model.parameters()).device

    optimizer = Adafactor(
        [p for p in model.parameters() if p.requires_grad],
        scale_parameter=False,
        relative_step=False,
        lr=1e-4,
        clip_threshold=1.0,
        weight_decay=1e-3,
    )
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)

    cleanup()
    losses: list[float] = []
    loss_rows: list[dict[str, object]] = []
    total_steps = 0

    # Preprocess once — preproc is deterministic, no need to redo every epoch
    for corpus in corpus_objects:
        corpus.df_train = corpus.df_train.copy()
        corpus.df_train['source_sentence'] = corpus.df_train['source_sentence'].apply(preproc)
        corpus.df_train['target_sentence'] = corpus.df_train['target_sentence'].apply(preproc)

    for epoch in range(num_epochs):
        # Re-sample every epoch so oversampled duplicates get fresh augmentations
        df_all, srcs, tgts = get_balanced_df(
            corpus_objects,
            temperature=cfg.sampling_temperature,
        )
        N = len(df_all)

        xx = df_all['source_sentence'].copy()
        yy = df_all['target_sentence'].copy()

        # Some additional data variation
        xx, yy = apply_variations(xx, yy)

        # Gronings-specific augmentation
        if np.any(tgts == 'gos_Latn'): # we should know where the gronings sentences are. TODO
            idxs = np.where(tgts == 'gos_Latn')[0]
            yy_idxs = list(yy[i] for i in idxs)
            yy_vals = add_gronings_variations(yy_idxs)
            yy_syns = swap_synonyms(yy_vals, synonym_pairs_gos)
            for k, i in enumerate(idxs):
                yy[i] = yy_syns[k]
        if np.any(srcs == 'gos_Latn'):
            idxs = np.where(srcs == 'gos_Latn')[0]
            xx_idxs = list(xx[i] for i in idxs)
            xx_vals = add_gronings_variations(xx_idxs)
            xx_syns = swap_synonyms(xx_vals, synonym_pairs_gos)
            for k, i in enumerate(idxs):
                xx[i] = xx_syns[k]

        # Randomly swap source and target languages
        swap_idxs = np.random.permutation(N)
        half = N // 2

        swap_mask = np.zeros(N, dtype=bool)
        swap_mask[:half] = True
        np.random.shuffle(swap_mask)

        xx_swapped = np.where(swap_mask, yy[swap_idxs], xx[swap_idxs])
        yy_swapped = np.where(swap_mask, xx[swap_idxs], yy[swap_idxs])
        src_swapped = np.where(swap_mask, tgts[swap_idxs], srcs[swap_idxs])
        tgt_swapped = np.where(swap_mask, srcs[swap_idxs], tgts[swap_idxs])

        # Shuffle
        final_idxs = np.random.permutation(N)
        df_all_aug = pd.DataFrame({
            "source_sentence": xx_swapped[final_idxs],
            "target_sentence": yy_swapped[final_idxs],
            "src_lang": src_swapped[final_idxs],
            "tgt_lang": tgt_swapped[final_idxs],
        })
        df_epoch = df_all_aug.sample(frac=1).reset_index(drop=True)
        # Bulk pre-tokenize all epoch data
        xx_texts = df_epoch['source_sentence'].tolist()
        yy_texts = df_epoch['target_sentence'].tolist()
        src_langs_epoch = df_epoch['src_lang'].tolist()
        tgt_langs_epoch = df_epoch['tgt_lang'].tolist()

        xx_input_ids, xx_attention = tokenize_mixed_langs(tokenizer, xx_texts, src_langs_epoch, max_length, device)
        yy_input_ids, yy_attention = tokenize_mixed_langs(tokenizer, yy_texts, tgt_langs_epoch, max_length, device)
        yy_input_ids[yy_input_ids == tokenizer.pad_token_id] = -100  # Masked loss targets

        n_samples_total = len(df_epoch)
        n_batches = int(np.ceil(n_samples_total / batch_size))
        tq = trange(n_batches, desc=f"Epoch {epoch+1}/{num_epochs}")
        for step in tq:
            batch_start = step * batch_size
            batch_end = min((step+1)*batch_size, n_samples_total)

            x = {
                "input_ids": xx_input_ids[batch_start:batch_end],
                "attention_mask": xx_attention[batch_start:batch_end]
            }
            y_input_ids_batch = yy_input_ids[batch_start:batch_end]
            loss = model(**x, labels=y_input_ids_batch).loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            loss_value = float(loss.item())
            losses.append(loss_value)
            loss_rows.append({"step": total_steps, "epoch": epoch + 1, "loss": loss_value})
            tq.set_postfix({'loss': np.mean(losses[-25:])})
            total_steps += 1

        print(f"Saving after epoch {epoch+1}")
        # Save checkpoints
        epoch_dir = checkpoints_dir / f"epoch{epoch+1}"
        model.save_pretrained(str(epoch_dir))
        tokenizer.save_pretrained(str(epoch_dir))
        cleanup()

    # Plotting and saving the losses
    plt.figure(figsize=(10, 5))
    pd.Series(losses).plot(label='Mean Loss')
    pd.Series(losses).ewm(span=30).mean().plot(label='Exponentially weighted moving average, 30 steps')
    pd.Series(losses).ewm(span=100).mean().plot(label='Exponentially weighted moving average, 100 steps')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.legend()

    # Save and plot loss history
    write_loss_csv(train_dir / "loss.csv", loss_rows)

    # Save the plot as an image
    plt.savefig(str(train_dir / "loss.png"))
    plt.close()

def main_train(corpus_objects: list, cfg: RunConfig | None = None):
    cfg = cfg or get_default_config()
    set_seed(cfg.seed)

    # Initialize run directory and persist run metadata once
    paths = init_run_dir(cfg.run_dir)
    run_config_dict = cfg.to_dict()
    write_json(paths["run_dir"] / "run_config.json", run_config_dict)
    (paths["run_dir"] / "run_config.txt").write_text(format_run_config_txt(run_config_dict), encoding="utf-8")

    model, tokenizer = setup_model_and_tokenizer(
        cfg.modelname,
        cfg.model_cache_path,
        cfg.new_lang_nllb,
        cfg.similar_lang_nllb,
        device=cfg.device,
    )
    train_model(model, tokenizer, corpus_objects, cfg)
