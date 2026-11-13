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
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
os.chdir(REPO_ROOT)

from nllb_try.config import RunConfig
from nllb_try.recipe import build_experiment, print_plan
from nllb_try.train import main_train

GEMMA_CORPUS = "data/kreuze/kreuze_synthetic_gemma50.csv"
TATOEBA_LANGS = ("nld", "gos", "eng", "deu", "spa")
NLLB_LANGS = ("nld_Latn", "gos_Latn", "eng_Latn", "deu_Latn", "spa_Latn")


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

    cfg = RunConfig(
        seed=args.seed,
        modelname=args.modelname,
        source_langs_tatoeba=TATOEBA_LANGS,
        source_langs_nllb=NLLB_LANGS,
        new_lang_nllb="gos_Latn",
        similar_lang_nllb="nld_Latn",
        tatoeba_path="data/tatoeba",
        parallel_data_paths=(GEMMA_CORPUS,),
        num_epochs=args.num_epochs,
        device=args.device,
    )
    corpora, cfg = build_experiment(args.variant, cfg)
    print_plan(f"Kreuze Phase 2: {args.variant}", corpora, cfg)
    if args.dry_run:
        return

    main_train(corpora, cfg)
    config_path = Path(cfg.run_dir) / "run_config.json"
    config = json.loads(config_path.read_text())
    config["experiment"] = f"kreuze-phase2-{args.variant}"
    config_path.write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    main()
