"""Run the controlled Phase 1 Gemma/Kreuze training experiment.

Both variants use Dutch and Gronings only, train for two epochs, alternate
translation direction between epochs, and process exactly the same number of
samples and optimizer steps.

Examples
--------
::

    uv run python run_kreuze_experiment.py --variant control --dry-run
    uv run python run_kreuze_experiment.py --variant pooled --dry-run
    uv run python run_kreuze_experiment.py --variant control --device cuda:0
    uv run python run_kreuze_experiment.py --variant pooled --device cuda:0
    uv run python run_kreuze_experiment.py --variant pooled \
        --modelname facebook/nllb-200-distilled-1.3B --device cuda:0
"""

import argparse
import json
import os
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

os.chdir(Path(__file__).resolve().parent)

from src.nllb_try.config import RunConfig
from src.nllb_try.corpus import (
    ParallelFileCorpus,
    TatoebaCorpus,
    main_corpus,
    pool_parallel_data_into_tatoeba,
)
from src.nllb_try.train import get_training_budget, main_train

GEMMA_CORPUS = "data/kreuze/kreuze_synthetic_gemma50.csv"


def build_experiment(
    variant: str, device: str, modelname: str
) -> tuple[list, RunConfig]:
    """Build one Phase 1 variant and its equalized training configuration."""
    timestamp = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d-%H%M%S")
    cfg = RunConfig(
        modelname=modelname,
        source_langs_tatoeba=("nld", "gos"),
        source_langs_nllb=("nld_Latn", "gos_Latn"),
        new_lang_nllb="gos_Latn",
        similar_lang_nllb="nld_Latn",
        tatoeba_path="data/tatoeba",
        parallel_data_paths=(GEMMA_CORPUS,),
        model_cache_path="hfacemodels",
        batch_size=256,
        num_epochs=2,
        warmup_steps=70,
        max_length=48,
        sampling_strategy="temperature",
        sampling_temperature=1.0,
        direction_strategy="alternating",
        device=device,
        run_id=f"kreuze-phase1-{variant}-{timestamp}",
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
    if len(tatoeba) != 1 or len(parallel) != 1:
        raise RuntimeError(
            "Phase 1 expects exactly one Tatoeba and one parallel corpus, "
            f"found {len(tatoeba)} and {len(parallel)}."
        )

    pooled_size = len(tatoeba[0].df_train) + len(parallel[0].df_train)
    cfg = replace(cfg, target_samples_per_epoch=pooled_size)

    if variant == "control":
        corpora = tatoeba
    elif variant == "pooled":
        corpora = pool_parallel_data_into_tatoeba(all_corpora)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    return corpora, cfg


def print_plan(variant: str, corpora: list, cfg: RunConfig) -> None:
    budget = get_training_budget(corpora, cfg)
    print(f"\nKreuze Phase 1: {variant}")
    print(f"  Model:             {cfg.modelname}")
    print(f"  Direction strategy: {cfg.direction_strategy}")
    print(f"  Epochs:            {cfg.num_epochs}")
    print(f"  Samples/epoch:     {budget['samples_per_epoch']:,}")
    print(f"  Steps/epoch:       {budget['steps_per_epoch']:,}")
    print(f"  Total steps:       {budget['total_steps']:,}")
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
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the corpora and print the exact budget without loading a model.",
    )
    args = parser.parse_args()

    corpora, cfg = build_experiment(args.variant, args.device, args.modelname)
    print_plan(args.variant, corpora, cfg)
    if args.dry_run:
        return

    main_train(corpora, cfg)

    config_path = Path(cfg.run_dir) / "run_config.json"
    config = json.loads(config_path.read_text())
    config["experiment"] = f"kreuze-phase1-{args.variant}"
    config_path.write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    main()
