import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm.auto import trange
from transformers import Adafactor, get_constant_schedule_with_warmup

from .artifacts import (
    format_run_config_txt,
    init_run_dir,
    write_json,
)
from .augmentation import (
    add_gronings_variations,
    apply_variations,
    preproc,
    swap_synonyms,
    synonym_pairs_gos,
)
from .config import RunConfig, get_default_config
from .seed import set_seed
from .tokenizer_and_model_setup import cleanup, setup_model_and_tokenizer


def _find_focus_corpus_index(
    corpus_objects: list, focus_lang_pair: tuple[str, str] | None
) -> int:
    if focus_lang_pair is None:
        raise ValueError(
            "focus_lang_pair must be set when sampling_strategy='focus_cap'"
        )

    focus_key = frozenset(focus_lang_pair)
    matches = [
        i
        for i, corpus in enumerate(corpus_objects)
        if frozenset((corpus.source_lang_nllb, corpus.target_lang_nllb)) == focus_key
    ]

    if not matches:
        raise ValueError(
            f"Focus pair {tuple(focus_lang_pair)} not found in the corpora"
        )
    if len(matches) > 1:
        raise ValueError(
            f"Focus pair {tuple(focus_lang_pair)} matched multiple corpora, expected one"
        )
    return matches[0]


def _get_temperature_sampling(
    corpus_objects: list,
    temperature: float,
    target_total_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    counts = np.array([len(c.df_train) for c in corpus_objects], dtype=float)
    probs = counts ** (1.0 / temperature)
    probs /= probs.sum()

    if target_total_samples is None:
        target_total_samples = int(counts.sum())

    sample_counts = np.maximum((probs * target_total_samples).astype(int), 1)
    return probs, sample_counts


def _get_focus_cap_sampling(
    corpus_objects: list, focus_lang_pair: tuple[str, str] | None
) -> tuple[np.ndarray, int]:
    counts = np.array([len(c.df_train) for c in corpus_objects], dtype=int)
    focus_idx = _find_focus_corpus_index(corpus_objects, focus_lang_pair)
    focus_size = counts[focus_idx]
    sample_counts = np.minimum(counts, focus_size)
    sample_counts[focus_idx] = focus_size
    return sample_counts, focus_idx


def get_balanced_df(
    corpus_objects: list,
    temperature: float = 5.0,
    sampling_strategy: str = "temperature",
    focus_lang_pair: tuple[str, str] | None = None,
    target_total_samples: int | None = None,
    verbose: bool = False,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Sample from each corpus using the configured balancing strategy.

    ``temperature`` reproduces the current temperature-based multilingual
    balancing.
    With *T=1* you get proportional sampling (status quo).
    With *T → ∞* every corpus contributes equally.
    *T=5* is a standard middle-ground for multilingual MT (NLLB / M2M-100)

    ``focus_cap`` instead keeps one configured focus pair at full
    size and caps every other pair at that same size, with fresh per-epoch
    sampling for the larger corpora

    Returns (df, src_langs, tgt_langs) where *df* has columns
    ``source_sentence``, ``target_sentence``, ``corpus_idx``, and the internal
    ``_row_index`` used by alternating-direction training.
    """
    dfs: list[pd.DataFrame] = []
    src_langs_all: list[str] = []
    tgt_langs_all: list[str] = []

    if sampling_strategy == "temperature":
        probs, sample_counts = _get_temperature_sampling(
            corpus_objects,
            temperature=temperature,
            target_total_samples=target_total_samples,
        )
        if verbose:
            print(f"Balanced sampling with temperature={temperature}")

        for i, corpus in enumerate(corpus_objects):
            n_samples = int(sample_counts[i])

            replace = n_samples > len(corpus.df_train)
            sampled = corpus.df_train.sample(n=n_samples, replace=replace)
            sampled = sampled.copy()
            sampled["_row_index"] = sampled.index.to_numpy()
            sampled["corpus_idx"] = i
            dfs.append(sampled)
            src_langs_all.extend([corpus.source_lang_nllb] * n_samples)
            tgt_langs_all.extend([corpus.target_lang_nllb] * n_samples)

            if verbose:
                ratio = n_samples / len(corpus.df_train)
                pct = probs[i] * 100
                direction = (
                    "oversampled"
                    if ratio > 1
                    else "undersampled"
                    if ratio < 1
                    else "exact"
                )
                print(
                    f"  Corpus {i} ({corpus.source_lang_nllb}→{corpus.target_lang_nllb}): "
                    f"{len(corpus.df_train):,} original → {n_samples:,} sampled "
                    f"({pct:.1f}%, {ratio:.2f}x, {direction})"
                )
    elif sampling_strategy == "focus_cap":
        if target_total_samples is not None:
            raise ValueError(
                "target_total_samples is not supported with sampling_strategy="
                "'focus_cap'; its per-corpus caps determine the epoch size."
            )
        sample_counts, focus_idx = _get_focus_cap_sampling(
            corpus_objects, focus_lang_pair
        )
        focus_corpus = corpus_objects[focus_idx]
        if verbose:
            print(
                "Focused sampling without temperature balancing "
                f"(focus={focus_corpus.source_lang_nllb}↔{focus_corpus.target_lang_nllb}, "
                f"reference size={sample_counts[focus_idx]:,})"
            )

        for i, corpus in enumerate(corpus_objects):
            n_samples = int(sample_counts[i])

            replace = n_samples > len(corpus.df_train)
            sampled = corpus.df_train.sample(n=n_samples, replace=replace)
            sampled = sampled.copy()
            sampled["_row_index"] = sampled.index.to_numpy()
            sampled["corpus_idx"] = i
            dfs.append(sampled)
            src_langs_all.extend([corpus.source_lang_nllb] * n_samples)
            tgt_langs_all.extend([corpus.target_lang_nllb] * n_samples)

            if verbose:
                ratio = n_samples / len(corpus.df_train)
                status = (
                    "focus" if i == focus_idx else "capped" if ratio < 1 else "kept"
                )
                print(
                    f"  Corpus {i} ({corpus.source_lang_nllb}→{corpus.target_lang_nllb}): "
                    f"{len(corpus.df_train):,} original → {n_samples:,} sampled "
                    f"({ratio:.2f}x, {status})"
                )
    else:
        raise ValueError(f"Unknown sampling_strategy: {sampling_strategy}")

    df = pd.concat(dfs).reset_index(drop=True)
    return (
        df,
        np.array(src_langs_all, dtype=object),
        np.array(tgt_langs_all, dtype=object),
    )


def get_training_budget(
    corpus_objects: list, cfg: RunConfig
) -> dict[str, int | list[int]]:
    """Calculate the exact sample and optimizer-step budget for a run."""
    if cfg.sampling_strategy == "temperature":
        _, sample_counts = _get_temperature_sampling(
            corpus_objects,
            temperature=cfg.sampling_temperature,
            target_total_samples=cfg.target_samples_per_epoch,
        )
    elif cfg.sampling_strategy == "focus_cap":
        if cfg.target_samples_per_epoch is not None:
            raise ValueError(
                "target_samples_per_epoch is not supported with "
                "sampling_strategy='focus_cap'."
            )
        sample_counts, _ = _get_focus_cap_sampling(corpus_objects, cfg.focus_lang_pair)
    else:
        raise ValueError(f"Unknown sampling_strategy: {cfg.sampling_strategy}")

    samples_per_epoch = int(sample_counts.sum())
    steps_per_epoch = math.ceil(samples_per_epoch / cfg.batch_size)
    return {
        "sample_counts": [int(count) for count in sample_counts],
        "samples_per_epoch": samples_per_epoch,
        "steps_per_epoch": steps_per_epoch,
        "total_samples": samples_per_epoch * cfg.num_epochs,
        "total_steps": steps_per_epoch * cfg.num_epochs,
    }


def _get_direction_mask(
    df: pd.DataFrame,
    permutation: np.ndarray,
    epoch: int,
    strategy: str,
) -> np.ndarray:
    """Choose which sampled pairs are reversed for one training epoch."""
    if strategy == "random":
        mask = np.zeros(len(df), dtype=bool)
        mask[: len(df) // 2] = True
        np.random.shuffle(mask)
        return mask
    if strategy == "alternating":
        row_indices = df["_row_index"].to_numpy()[permutation]
        corpus_indices = df["corpus_idx"].to_numpy()[permutation]
        return (row_indices + corpus_indices + epoch) % 2 == 1
    raise ValueError(f"Unknown direction_strategy: {strategy}")


def _get_model_initialization(cfg: RunConfig) -> tuple[str, str | None]:
    """Choose the model source without reinitializing a resumed language token."""
    if cfg.initial_model_path is not None:
        return cfg.initial_model_path, None
    return cfg.modelname, cfg.similar_lang_nllb


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
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        for j, i_global in enumerate(idxs):
            input_ids_dict[i_global] = feats["input_ids"][j]
            attention_mask_dict[i_global] = feats["attention_mask"][j]
    input_ids: list[torch.Tensor] = [input_ids_dict[i] for i in range(len(texts))]
    attention_mask: list[torch.Tensor] = [
        attention_mask_dict[i] for i in range(len(texts))
    ]
    input_ids_tensor = torch.stack(input_ids).to(device)
    attention_mask_tensor = torch.stack(attention_mask).to(device)
    return input_ids_tensor, attention_mask_tensor


def train_model(
    model, tokenizer, corpus_objects: list, cfg: RunConfig, verbose: bool = True
) -> None:
    batch_size: int = cfg.batch_size
    max_length: int = cfg.max_length
    num_epochs: int = cfg.num_epochs
    warmup_steps: int = cfg.warmup_steps

    paths = init_run_dir(cfg.run_dir)
    checkpoints_dir = paths["checkpoints_dir"]
    train_dir = paths["train_dir"]

    device = next(model.parameters()).device

    # ── Verbose: show training config ──
    if verbose:
        total_original = sum(len(c.df_train) for c in corpus_objects)
        budget = get_training_budget(corpus_objects, cfg)
        sample_counts = budget["sample_counts"]
        total_sampled = budget["samples_per_epoch"]
        n_batches = budget["steps_per_epoch"]

        print(f"\n{'=' * 65}")
        print("  Training plan")
        print(f"{'=' * 65}")
        print(f"  Model:        {cfg.modelname}")
        print(f"  Epochs:       {num_epochs}")
        print(f"  Batch size:   {batch_size}")
        print(f"  Learning rate:{cfg.learning_rate}")
        print(f"  Max length:   {max_length} tokens")
        print(f"  Warmup:       {warmup_steps} steps")
        print(f"  Sampling:     {cfg.sampling_strategy}")
        print(f"  Directions:   {cfg.direction_strategy}")
        if cfg.sampling_strategy == "temperature":
            print(f"  Temperature:  {cfg.sampling_temperature}")
            if cfg.target_samples_per_epoch is not None:
                print(f"  Epoch target: {cfg.target_samples_per_epoch:,} samples")
        else:
            focus_idx = _find_focus_corpus_index(corpus_objects, cfg.focus_lang_pair)
            focus_corpus = corpus_objects[focus_idx]
            print(
                "  Focus pair:   "
                f"{focus_corpus.source_lang_nllb}↔{focus_corpus.target_lang_nllb}"
            )
            print(f"  Focus size:   {sample_counts[focus_idx]:,}")
        print(f"  Device:       {device}")
        print(f"  Original rows:{total_original:,}")
        print(f"  Sampled rows: {total_sampled:,}")
        print(f"  Steps/epoch:  {n_batches:,}")
        print(f"  Total steps:  {budget['total_steps']:,}")
        print(f"{'=' * 65}\n")

    optimizer = Adafactor(
        [p for p in model.parameters() if p.requires_grad],
        scale_parameter=False,
        relative_step=False,
        lr=cfg.learning_rate,
        clip_threshold=1.0,
        weight_decay=1e-3,
    )
    scheduler = get_constant_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps
    )

    cleanup()
    losses: list[float] = []
    loss_rows: list[dict[str, object]] = []
    total_steps = 0

    # Preprocess once — preproc is deterministic, no need to redo every epoch
    for corpus in corpus_objects:
        corpus.df_train = corpus.df_train.copy()
        corpus.df_train["source_sentence"] = corpus.df_train["source_sentence"].apply(
            preproc
        )
        corpus.df_train["target_sentence"] = corpus.df_train["target_sentence"].apply(
            preproc
        )

    for epoch in range(num_epochs):
        # Re-sample every epoch so oversampled duplicates get fresh augmentations
        df_all, srcs, tgts = get_balanced_df(
            corpus_objects,
            temperature=cfg.sampling_temperature,
            sampling_strategy=cfg.sampling_strategy,
            focus_lang_pair=cfg.focus_lang_pair,
            target_total_samples=cfg.target_samples_per_epoch,
            verbose=verbose
            and epoch == 0,  # only print sampling details for first epoch
        )
        N = len(df_all)

        xx = df_all["source_sentence"].copy()
        yy = df_all["target_sentence"].copy()

        # Some additional data variation
        xx, yy = apply_variations(xx, yy)

        # Gronings-specific augmentation
        if np.any(
            tgts == "gos_Latn"
        ):  # we should know where the gronings sentences are. TODO
            idxs = np.where(tgts == "gos_Latn")[0]
            yy_idxs = [yy[i] for i in idxs]
            yy_vals = add_gronings_variations(yy_idxs)
            yy_syns = swap_synonyms(yy_vals, synonym_pairs_gos)
            for k, i in enumerate(idxs):
                yy[i] = yy_syns[k]
        if np.any(srcs == "gos_Latn"):
            idxs = np.where(srcs == "gos_Latn")[0]
            xx_idxs = [xx[i] for i in idxs]
            xx_vals = add_gronings_variations(xx_idxs)
            xx_syns = swap_synonyms(xx_vals, synonym_pairs_gos)
            for k, i in enumerate(idxs):
                xx[i] = xx_syns[k]

        # Choose a direction for every sampled pair.
        swap_idxs = np.random.permutation(N)
        swap_mask = _get_direction_mask(
            df_all,
            permutation=swap_idxs,
            epoch=epoch,
            strategy=cfg.direction_strategy,
        )

        xx_swapped = np.where(swap_mask, yy[swap_idxs], xx[swap_idxs])
        yy_swapped = np.where(swap_mask, xx[swap_idxs], yy[swap_idxs])
        src_swapped = np.where(swap_mask, tgts[swap_idxs], srcs[swap_idxs])
        tgt_swapped = np.where(swap_mask, srcs[swap_idxs], tgts[swap_idxs])

        # Shuffle
        final_idxs = np.random.permutation(N)
        df_all_aug = pd.DataFrame(
            {
                "source_sentence": xx_swapped[final_idxs],
                "target_sentence": yy_swapped[final_idxs],
                "src_lang": src_swapped[final_idxs],
                "tgt_lang": tgt_swapped[final_idxs],
            }
        )
        df_epoch = df_all_aug.sample(frac=1).reset_index(drop=True)
        # Bulk pre-tokenize all epoch data
        xx_texts = df_epoch["source_sentence"].tolist()
        yy_texts = df_epoch["target_sentence"].tolist()
        src_langs_epoch = df_epoch["src_lang"].tolist()
        tgt_langs_epoch = df_epoch["tgt_lang"].tolist()

        xx_input_ids, xx_attention = tokenize_mixed_langs(
            tokenizer, xx_texts, src_langs_epoch, max_length, device
        )
        yy_input_ids, _ = tokenize_mixed_langs(
            tokenizer, yy_texts, tgt_langs_epoch, max_length, device
        )
        yy_input_ids[
            yy_input_ids == tokenizer.pad_token_id
        ] = -100  # Masked loss targets

        n_samples_total = len(df_epoch)
        n_batches = int(np.ceil(n_samples_total / batch_size))
        tq = trange(n_batches, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for step in tq:
            batch_start = step * batch_size
            batch_end = min((step + 1) * batch_size, n_samples_total)

            x = {
                "input_ids": xx_input_ids[batch_start:batch_end],
                "attention_mask": xx_attention[batch_start:batch_end],
            }
            y_input_ids_batch = yy_input_ids[batch_start:batch_end]
            loss = model(**x, labels=y_input_ids_batch).loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            loss_value = float(loss.item())
            losses.append(loss_value)
            loss_rows.append(
                {"step": total_steps, "epoch": epoch + 1, "loss": loss_value}
            )
            tq.set_postfix({"loss": np.mean(losses[-25:])})
            total_steps += 1

        print(f"Saving after epoch {epoch + 1}")
        # Save checkpoints
        epoch_dir = checkpoints_dir / f"epoch{epoch + 1}"
        model.save_pretrained(str(epoch_dir))
        tokenizer.save_pretrained(str(epoch_dir))
        cleanup()

    # Plotting and saving the losses
    plt.figure(figsize=(10, 5))
    pd.Series(losses).plot(label="Mean Loss")
    pd.Series(losses).ewm(span=30).mean().plot(
        label="Exponentially weighted moving average, 30 steps"
    )
    pd.Series(losses).ewm(span=100).mean().plot(
        label="Exponentially weighted moving average, 100 steps"
    )
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Time")
    plt.legend()

    # Save the plot as an image
    plt.savefig(str(train_dir / "loss.png"))
    plt.close()


def main_train(
    corpus_objects: list, cfg: RunConfig | None = None, verbose: bool = True
):
    cfg = cfg or get_default_config()
    set_seed(cfg.seed)

    # Initialize run directory and persist run metadata once
    paths = init_run_dir(cfg.run_dir)
    run_config_dict = cfg.to_dict()
    write_json(paths["run_dir"] / "run_config.json", run_config_dict)
    (paths["run_dir"] / "run_config.txt").write_text(
        format_run_config_txt(run_config_dict), encoding="utf-8"
    )
    if verbose:
        print("ron_config.json and run_config.txt saved")

    model_source, similar_lang = _get_model_initialization(cfg)
    model, tokenizer = setup_model_and_tokenizer(
        model_source,
        cfg.model_cache_path,
        cfg.new_lang_nllb,
        similar_lang,
        device=cfg.device,
    )
    train_model(model, tokenizer, corpus_objects, cfg, verbose=verbose)
