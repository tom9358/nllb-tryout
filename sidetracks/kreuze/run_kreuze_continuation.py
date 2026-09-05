"""Run the equal-step pooled-versus-clean Phase 1b continuation experiment."""

import argparse
import json
import os
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

from nllb_try.config import RunConfig
from nllb_try.corpus import (
    ParallelFileCorpus,
    TatoebaCorpus,
    main_corpus,
    pool_parallel_data_into_tatoeba,
)
from nllb_try.train import get_training_budget, main_train
from sidetracks.kreuze.run_kreuze_evaluation import find_latest_run


def build_continuation(
    variant: str, source_run: Path, device: str
) -> tuple[list, RunConfig]:
    source_config = json.loads((source_run / "run_config.json").read_text())
    source_checkpoint = source_run / "checkpoints" / "epoch2"
    if not source_checkpoint.is_dir():
        raise FileNotFoundError(f"Missing source checkpoint: {source_checkpoint}")

    timestamp = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d-%H%M%S")
    cfg = RunConfig(
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
        batch_size=256,
        learning_rate=1e-4,
        num_epochs=1,
        warmup_steps=0,
        max_length=48,
        sampling_strategy="temperature",
        sampling_temperature=1.0,
        direction_strategy="alternating",
        device=device,
        run_id=f"kreuze-phase1b-{variant}-{timestamp}",
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
            "Phase 1b expects exactly one Tatoeba and one parallel corpus, "
            f"found {len(tatoeba)} and {len(parallel)}."
        )

    cfg = replace(cfg, target_samples_per_epoch=len(tatoeba[0].df_train))
    if variant in {"clean", "control"}:
        corpora = tatoeba
    elif variant == "pooled":
        corpora = pool_parallel_data_into_tatoeba(all_corpora)
    else:
        raise ValueError(f"Unknown variant: {variant}")
    return corpora, cfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant", choices=("clean", "pooled", "control"), required=True
    )
    parser.add_argument("--source-run", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    source_variant = "control" if args.variant == "control" else "pooled"
    source_run = args.source_run or find_latest_run(source_variant)
    corpora, cfg = build_continuation(args.variant, source_run, args.device)
    budget = get_training_budget(corpora, cfg)
    print(f"\nKreuze Phase 1b: {args.variant}")
    print(f"  Source checkpoint: {cfg.initial_model_path}")
    print(f"  Learning rate:     {cfg.learning_rate}")
    print(f"  Samples:           {budget['samples_per_epoch']:,}")
    print(f"  Total steps:       {budget['total_steps']:,}")
    if args.dry_run:
        return

    main_train(corpora, cfg)
    config_path = Path(cfg.run_dir) / "run_config.json"
    config = json.loads(config_path.read_text())
    config["experiment"] = f"kreuze-phase1b-{args.variant}"
    config_path.write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    main()
