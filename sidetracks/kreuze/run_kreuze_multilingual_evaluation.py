"""Evaluate a five-language Phase 2 run on every validation corpus."""

import argparse
import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
os.chdir(REPO_ROOT)

from nllb_try.config import RunConfig
from nllb_try.corpus import main_corpus
from nllb_try.evaluate import main_evaluate

EVAL_ID = "kreuze-phase2-full-validation"


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
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--include-baseline", action="store_true")
    args = parser.parse_args()

    corpora, new_lang_nllb = build_validation_corpora(args.run_dir)
    print(f"Evaluating: {args.run_dir}")
    main_evaluate(
        corpus_objects=corpora,
        run_dir=str(args.run_dir),
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
