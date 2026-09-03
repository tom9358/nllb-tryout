"""Run an equal-step continuation of a five-language Phase 2 model."""

import argparse
import json
import os
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

os.chdir(Path(__file__).resolve().parent)

from run_kreuze_multilingual_experiment import (
    FOCUS_PAIR,
    _get_single_focus_corpus,
)
from src.nllb_try.config import RunConfig
from src.nllb_try.corpus import (
    BaseParallelCorpus,
    ParallelFileCorpus,
    TatoebaCorpus,
    main_corpus,
    pool_parallel_data_into_tatoeba,
)
from src.nllb_try.train import get_training_budget, main_train


def find_latest_run(variant: str, modelname: str) -> Path:
    candidates: list[Path] = []
    for path in Path("checkpoints").iterdir():
        config_path = path / "run_config.json"
        checkpoint = path / "checkpoints" / "epoch2"
        if not config_path.is_file() or not checkpoint.is_dir():
            continue
        config = json.loads(config_path.read_text())
        if (
            config.get("experiment") == f"kreuze-phase2-{variant}"
            and config.get("modelname") == modelname
        ):
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(
            f"No completed Phase 2 {variant} run found for {modelname}."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def get_final_checkpoint(source_run: Path, source_config: dict) -> Path:
    source_epoch = source_config["num_epochs"]
    source_checkpoint = source_run / "checkpoints" / f"epoch{source_epoch}"
    if not source_checkpoint.is_dir():
        raise FileNotFoundError(f"Missing source checkpoint: {source_checkpoint}")
    return source_checkpoint


def build_continuation(
    variant: str, source_run: Path, device: str
) -> tuple[list[BaseParallelCorpus], RunConfig]:
    source_config = json.loads((source_run / "run_config.json").read_text())
    source_checkpoint = get_final_checkpoint(source_run, source_config)

    expected_source = "control" if variant == "control" else "pooled"
    expected_experiment = f"kreuze-phase2-{expected_source}"
    if source_config.get("experiment") != expected_experiment:
        raise ValueError(
            f"{variant} continuation requires {expected_experiment}, found "
            f"{source_config.get('experiment')!r}."
        )

    timestamp = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d-%H%M%S")
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
        batch_size=256,
        learning_rate=1e-4,
        num_epochs=1,
        warmup_steps=0,
        max_length=48,
        sampling_strategy="focus_total",
        sampling_temperature=5.0,
        focus_lang_pair=FOCUS_PAIR,
        direction_strategy="alternating",
        device=device,
        run_id=(f"kreuze-phase2b-{variant}-seed{source_config['seed']}-{timestamp}"),
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
            "Phase 2b expects ten Tatoeba corpora and one parallel corpus, "
            f"found {len(tatoeba)} and {len(parallel)}."
        )

    clean_focus_size = len(_get_single_focus_corpus(tatoeba).df_train)
    cfg = replace(cfg, target_samples_per_epoch=2 * clean_focus_size)

    if variant in {"clean", "control"}:
        corpora: list[BaseParallelCorpus] = tatoeba
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
    parser.add_argument(
        "--source-modelname",
        default="facebook/nllb-200-distilled-600M",
        help="Used to find the source run when --source-run is omitted.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    source_variant = "control" if args.variant == "control" else "pooled"
    source_run = args.source_run or find_latest_run(
        source_variant, args.source_modelname
    )
    corpora, cfg = build_continuation(args.variant, source_run, args.device)
    budget = get_training_budget(corpora, cfg)
    print(f"\nKreuze Phase 2b: {args.variant}")
    print(f"  Source checkpoint: {cfg.initial_model_path}")
    print(f"  Learning rate:     {cfg.learning_rate}")
    print(f"  Samples:           {budget['samples_per_epoch']:,}")
    print(f"  Total steps:       {budget['total_steps']:,}")
    for corpus, sample_count in zip(corpora, budget["sample_counts"]):
        print(
            f"  {corpus.source_lang_nllb}↔{corpus.target_lang_nllb}: "
            f"{sample_count:,} samples"
        )
    if args.dry_run:
        return

    main_train(corpora, cfg)
    config_path = Path(cfg.run_dir) / "run_config.json"
    config = json.loads(config_path.read_text())
    config["experiment"] = f"kreuze-phase2b-{args.variant}"
    config["source_run"] = str(source_run)
    config_path.write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    main()
