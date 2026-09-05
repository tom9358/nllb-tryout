"""Run the controlled five-language Kreuze comparison.

Both variants use Dutch, Gronings, English, German and Spanish with
``focus_total`` sampling. Half of every epoch is Dutch-Gronings and half is
shared by all other language pairs. The Tatoeba-only control and pooled
Kreuze treatment receive exactly the same sample and optimizer-step budget.

Examples
--------
::

    uv run python sidetracks/kreuze/run_kreuze_multilingual_experiment.py \
        --variant control --dry-run
    uv run python sidetracks/kreuze/run_kreuze_multilingual_experiment.py \
        --variant pooled --dry-run
"""

import argparse
import json
import os
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
os.chdir(REPO_ROOT)

from nllb_try.config import RunConfig
from nllb_try.corpus import (
    BaseParallelCorpus,
    ParallelFileCorpus,
    TatoebaCorpus,
    main_corpus,
    pool_parallel_data_into_tatoeba,
)
from nllb_try.train import get_training_budget, main_train

GEMMA_CORPUS = "data/kreuze/kreuze_synthetic_gemma50.csv"
TATOEBA_LANGS = ("nld", "gos", "eng", "deu", "spa")
NLLB_LANGS = (
    "nld_Latn",
    "gos_Latn",
    "eng_Latn",
    "deu_Latn",
    "spa_Latn",
)
FOCUS_PAIR = ("nld_Latn", "gos_Latn")


def _matches_focus(corpus: BaseParallelCorpus) -> bool:
    return frozenset((corpus.source_lang_nllb, corpus.target_lang_nllb)) == frozenset(
        FOCUS_PAIR
    )


def _get_single_focus_corpus(
    corpora: list[BaseParallelCorpus],
) -> BaseParallelCorpus:
    matches = [corpus for corpus in corpora if _matches_focus(corpus)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one Dutch-Gronings corpus, found {len(matches)}.")
    return matches[0]


def _get_pooled_focus_size(corpora: list[BaseParallelCorpus]) -> int:
    matches = [corpus for corpus in corpora if _matches_focus(corpus)]
    if len(matches) != 2:
        raise RuntimeError(
            "Expected one Tatoeba and one parallel Dutch-Gronings corpus, "
            f"found {len(matches)}."
        )
    return sum(len(corpus.df_train) for corpus in matches)


def build_experiment(
    variant: str,
    device: str,
    modelname: str,
    seed: int = 9358,
    num_epochs: int = 2,
) -> tuple[list[BaseParallelCorpus], RunConfig]:
    """Build one equal-budget five-language training variant."""
    timestamp = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d-%H%M%S")
    cfg = RunConfig(
        seed=seed,
        modelname=modelname,
        source_langs_tatoeba=TATOEBA_LANGS,
        source_langs_nllb=NLLB_LANGS,
        new_lang_nllb="gos_Latn",
        similar_lang_nllb="nld_Latn",
        tatoeba_path="data/tatoeba",
        parallel_data_paths=(GEMMA_CORPUS,),
        model_cache_path="hfacemodels",
        batch_size=256,
        num_epochs=num_epochs,
        warmup_steps=70,
        max_length=48,
        sampling_strategy="focus_total",
        sampling_temperature=5.0,
        focus_lang_pair=FOCUS_PAIR,
        direction_strategy="alternating",
        device=device,
        run_id=f"kreuze-phase2-{variant}-seed{seed}-{timestamp}",
    )

    all_corpora = main_corpus(
        source_langs_tatoeba=cfg.source_langs_tatoeba,
        source_langs_nllb=cfg.source_langs_nllb,
        parallel_data_paths=cfg.parallel_data_paths,
        cfg=cfg,
    )
    tatoeba = [corpus for corpus in all_corpora if isinstance(corpus, TatoebaCorpus)]
    parallel = [
        corpus for corpus in all_corpora if isinstance(corpus, ParallelFileCorpus)
    ]
    if len(tatoeba) != 10 or len(parallel) != 1:
        raise RuntimeError(
            "Phase 2 expects ten Tatoeba corpora and one parallel corpus, "
            f"found {len(tatoeba)} and {len(parallel)}."
        )

    target_samples = 2 * _get_pooled_focus_size(all_corpora)
    cfg = replace(cfg, target_samples_per_epoch=target_samples)

    if variant == "control":
        corpora: list[BaseParallelCorpus] = tatoeba
    elif variant == "pooled":
        corpora = pool_parallel_data_into_tatoeba(all_corpora)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    return corpora, cfg


def print_plan(variant: str, corpora: list[BaseParallelCorpus], cfg: RunConfig) -> None:
    budget = get_training_budget(corpora, cfg)
    print(f"\nKreuze Phase 2: {variant}")
    print(f"  Model:              {cfg.modelname}")
    print(f"  Languages:          {', '.join(cfg.source_langs_nllb)}")
    print(
        f"  Sampling:           {cfg.sampling_strategy}, T={cfg.sampling_temperature:g}"
    )
    print(f"  Focus pair:         {'↔'.join(cfg.focus_lang_pair or ())}")
    print(f"  Direction strategy: {cfg.direction_strategy}")
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("control", "pooled"), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--modelname",
        default="facebook/nllb-200-distilled-600M",
        help="Hugging Face model name or local model path.",
    )
    parser.add_argument("--seed", type=int, default=9358)
    parser.add_argument("--num-epochs", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    corpora, cfg = build_experiment(
        args.variant,
        args.device,
        args.modelname,
        seed=args.seed,
        num_epochs=args.num_epochs,
    )
    print_plan(args.variant, corpora, cfg)
    if args.dry_run:
        return

    main_train(corpora, cfg)
    config_path = Path(cfg.run_dir) / "run_config.json"
    config = json.loads(config_path.read_text())
    config["experiment"] = f"kreuze-phase2-{args.variant}"
    config_path.write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    main()
