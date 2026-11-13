"""Training recipe of the released model (issue #34).

1. A *pooled* stage: all Tatoeba pairs between the configured languages, with
   the extra parallel data (e.g. synthetic Kreuze sentences) pooled into the
   focus pair. ``focus_total`` sampling gives every epoch 50% focus pair and
   50% the other pairs at temperature 5.
2. A *clean* continuation of one epoch from a pooled checkpoint, on Tatoeba
   only.

Languages, model, seed, epochs and data paths come from the caller's
``RunConfig``; the focus pair is ``(similar_lang_nllb, new_lang_nllb)``.
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from .config import RunConfig
from .corpus import (
    BaseParallelCorpus,
    ParallelFileCorpus,
    TatoebaCorpus,
    main_corpus,
    pool_parallel_data_into_tatoeba,
)
from .train import get_training_budget

RECIPE = dict(
    batch_size=256,
    max_length=48,
    sampling_strategy="focus_total",
    sampling_temperature=5.0,
    direction_strategy="alternating",
)


def focus_pair(cfg: RunConfig) -> tuple[str, str]:
    return (cfg.similar_lang_nllb, cfg.new_lang_nllb)


def _focus_corpora(
    corpora: list[BaseParallelCorpus], pair: tuple[str, str]
) -> list[BaseParallelCorpus]:
    key = frozenset(pair)
    return [c for c in corpora if frozenset((c.source_lang_nllb, c.target_lang_nllb)) == key]


def _get_single_focus_corpus(
    corpora: list[BaseParallelCorpus], pair: tuple[str, str]
) -> BaseParallelCorpus:
    matches = _focus_corpora(corpora, pair)
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {'-'.join(pair)} corpus, found {len(matches)}.")
    return matches[0]


def _get_pooled_focus_size(
    corpora: list[BaseParallelCorpus], pair: tuple[str, str]
) -> int:
    """Training rows of the focus pair once all its corpora are pooled."""
    return sum(len(c.df_train) for c in _focus_corpora(corpora, pair))


def _load_corpora(
    cfg: RunConfig,
) -> tuple[list[BaseParallelCorpus], list[TatoebaCorpus], list[ParallelFileCorpus]]:
    if len(cfg.source_langs_tatoeba) != len(cfg.source_langs_nllb):
        raise ValueError("source_langs_tatoeba and source_langs_nllb must have equal length.")
    pair = focus_pair(cfg)
    if not set(pair) <= set(cfg.source_langs_nllb):
        raise ValueError(f"Focus pair {pair} must be among source_langs_nllb.")

    all_corpora = main_corpus(
        source_langs_tatoeba=cfg.source_langs_tatoeba,
        source_langs_nllb=cfg.source_langs_nllb,
        parallel_data_paths=cfg.parallel_data_paths,
        parallel_data_separator=cfg.parallel_data_separator,
        cfg=cfg,
    )
    tatoeba = [c for c in all_corpora if isinstance(c, TatoebaCorpus)]
    parallel = [c for c in all_corpora if isinstance(c, ParallelFileCorpus)]
    n = len(cfg.source_langs_tatoeba)
    if len(tatoeba) != n * (n - 1) // 2:
        raise RuntimeError(
            f"Expected {n * (n - 1) // 2} Tatoeba corpora for {n} languages, found {len(tatoeba)}."
        )
    if cfg.parallel_data_paths and not parallel:
        raise RuntimeError(f"No parallel data found in {cfg.parallel_data_paths}.")
    return all_corpora, tatoeba, parallel


def _timestamp() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y%m%d-%H%M%S")


def build_experiment(
    variant: str, cfg: RunConfig
) -> tuple[list[BaseParallelCorpus], RunConfig]:
    """Build the pooled stage (``variant="pooled"``) or its Tatoeba-only control.

    Both variants get the same sample budget: twice the pooled focus size.
    """
    if variant not in {"control", "pooled"}:
        raise ValueError(f"Unknown variant: {variant}")
    cfg = replace(
        cfg,
        **RECIPE,
        warmup_steps=70,
        focus_lang_pair=focus_pair(cfg),
        run_id=f"kreuze-phase2-{variant}-seed{cfg.seed}-{_timestamp()}",
    )
    all_corpora, tatoeba, parallel = _load_corpora(cfg)
    if not parallel:
        raise RuntimeError("The recipe needs extra parallel data for the focus pair.")

    cfg = replace(
        cfg,
        target_samples_per_epoch=2 * _get_pooled_focus_size(all_corpora, cfg.focus_lang_pair),
    )
    if variant == "control":
        return list(tatoeba), cfg
    return pool_parallel_data_into_tatoeba(all_corpora), cfg


def get_source_checkpoint(
    source_run: Path, source_config: dict, source_epoch: int | None = None
) -> tuple[Path, int]:
    final_epoch = source_config["num_epochs"]
    source_epoch = final_epoch if source_epoch is None else source_epoch
    if not 1 <= source_epoch <= final_epoch:
        raise ValueError(
            f"source_epoch must be between 1 and {final_epoch}, got {source_epoch}."
        )
    source_checkpoint = source_run / "checkpoints" / f"epoch{source_epoch}"
    if not source_checkpoint.is_dir():
        raise FileNotFoundError(f"Missing source checkpoint: {source_checkpoint}")
    return source_checkpoint, source_epoch


def build_continuation(
    variant: str, source_run: Path, device: str, source_epoch: int | None = None
) -> tuple[list[BaseParallelCorpus], RunConfig]:
    """One-epoch continuation of a ``build_experiment`` run.

    ``clean`` and ``control`` train on Tatoeba only; ``pooled`` keeps the extra
    parallel data. The budget is twice the Tatoeba focus size, so the focus half
    is one full pass over the Tatoeba focus pair. Languages and data paths are
    taken from the source run.
    """
    source_run = Path(source_run)
    source_config = json.loads((source_run / "run_config.json").read_text())
    source_checkpoint, source_epoch = get_source_checkpoint(
        source_run, source_config, source_epoch
    )

    expected_experiment = f"kreuze-phase2-{'control' if variant == 'control' else 'pooled'}"
    if source_config.get("experiment") != expected_experiment:
        raise ValueError(
            f"{variant} continuation requires {expected_experiment}, found "
            f"{source_config.get('experiment')!r}."
        )

    cfg = RunConfig(
        seed=source_config["seed"],
        modelname=source_config["modelname"],
        initial_model_path=str(source_checkpoint),
        source_langs_tatoeba=tuple(source_config["source_langs_tatoeba"]),
        source_langs_nllb=tuple(source_config["source_langs_nllb"]),
        new_lang_nllb=source_config["new_lang_nllb"],
        similar_lang_nllb=source_config["similar_lang_nllb"],
        tatoeba_path=source_config["tatoeba_path"],
        parallel_data_paths=tuple(source_config["parallel_data_paths"]),
        parallel_data_separator=source_config.get("parallel_data_separator"),
        model_cache_path=source_config["model_cache_path"],
        **RECIPE,
        learning_rate=1e-4,
        num_epochs=1,
        warmup_steps=0,
        direction_epoch_offset=source_epoch,
        device=device,
        run_id=(
            f"kreuze-phase2b-{variant}-from-epoch{source_epoch}-"
            f"seed{source_config['seed']}-{_timestamp()}"
        ),
    )
    cfg = replace(cfg, focus_lang_pair=focus_pair(cfg))
    all_corpora, tatoeba, _ = _load_corpora(cfg)

    clean_focus_size = len(_get_single_focus_corpus(tatoeba, cfg.focus_lang_pair).df_train)
    cfg = replace(cfg, target_samples_per_epoch=2 * clean_focus_size)

    if variant in {"clean", "control"}:
        return list(tatoeba), cfg
    if variant == "pooled":
        return pool_parallel_data_into_tatoeba(all_corpora), cfg
    raise ValueError(f"Unknown variant: {variant}")


def print_plan(label: str, corpora: list[BaseParallelCorpus], cfg: RunConfig) -> None:
    budget = get_training_budget(corpora, cfg)
    print(f"\n{label}")
    print(f"  Model:              {cfg.initial_model_path or cfg.modelname}")
    print(f"  Languages:          {', '.join(cfg.source_langs_nllb)}")
    print(
        f"  Sampling:           {cfg.sampling_strategy}, T={cfg.sampling_temperature:g}"
    )
    print(f"  Focus pair:         {'↔'.join(cfg.focus_lang_pair or ())}")
    print(f"  Direction strategy: {cfg.direction_strategy}")
    print(f"  Learning rate:      {cfg.learning_rate}")
    print(f"  Epochs:             {cfg.num_epochs}")
    print(f"  Samples/epoch:      {budget['samples_per_epoch']:,}")
    print(f"  Steps/epoch:        {budget['steps_per_epoch']:,}")
    print(f"  Total steps:        {budget['total_steps']:,}")
    for corpus, sample_count in zip(corpora, budget["sample_counts"]):
        print(
            f"  {corpus.source_lang_nllb}↔{corpus.target_lang_nllb}: "
            f"{len(corpus.df_train):,} unique train rows, "
            f"{sample_count:,} samples/epoch"
        )
