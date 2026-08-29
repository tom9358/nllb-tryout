"""Evaluate a completed Kreuze Phase 1 run on full validation splits."""

import argparse
import json
import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parent)

from src.nllb_try.config import RunConfig
from src.nllb_try.corpus import main_corpus
from src.nllb_try.evaluate import main_evaluate

EVAL_ID = "kreuze-phase1-full-validation"


def find_latest_run(variant: str) -> Path:
    pattern = f"nllb-200-distilled-600M-nld-gos-kreuze-phase1-{variant}-*"
    candidates = [
        path
        for path in Path("checkpoints").glob(pattern)
        if (path / "checkpoints" / "epoch2").is_dir()
    ]
    if not candidates:
        raise FileNotFoundError(f"No completed Phase 1 {variant} run found.")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def build_validation_corpora(run_dir: Path):
    run_config = json.loads((run_dir / "run_config.json").read_text())
    cfg = RunConfig(
        source_langs_tatoeba=tuple(run_config["source_langs_tatoeba"]),
        source_langs_nllb=tuple(run_config["source_langs_nllb"]),
        tatoeba_path=run_config["tatoeba_path"],
        parallel_data_paths=tuple(run_config["parallel_data_paths"]),
        parallel_data_separator=run_config.get("parallel_data_separator"),
    )
    corpora = main_corpus(
        source_langs_tatoeba=cfg.source_langs_tatoeba,
        source_langs_nllb=cfg.source_langs_nllb,
        parallel_data_paths=cfg.parallel_data_paths,
        cfg=cfg,
    )
    return corpora, run_config["new_lang_nllb"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("control", "pooled"), required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--include-baseline", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir or find_latest_run(args.variant)
    corpora, new_lang_nllb = build_validation_corpora(run_dir)
    print(f"Evaluating {args.variant}: {run_dir}")
    main_evaluate(
        corpus_objects=corpora,
        run_dir=str(run_dir),
        new_lang_nllb=new_lang_nllb,
        eval_id=EVAL_ID,
        device=args.device,
        sample_size=None,
        batch_size=128,
        include_baseline=args.include_baseline,
        include_train=False,
    )


if __name__ == "__main__":
    main()
