"""Run an equal-step continuation of a five-language Phase 2 model."""

import argparse
import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
os.chdir(REPO_ROOT)

from nllb_try.recipe import build_continuation, print_plan
from nllb_try.train import main_train


def find_latest_run(variant: str, modelname: str) -> Path:
    candidates: list[Path] = []
    for path in Path("checkpoints").iterdir():
        config_path = path / "run_config.json"
        if not config_path.is_file():
            continue
        config = json.loads(config_path.read_text())
        checkpoint = path / "checkpoints" / f"epoch{config['num_epochs']}"
        if (
            checkpoint.is_dir()
            and config.get("experiment") == f"kreuze-phase2-{variant}"
            and config.get("modelname") == modelname
        ):
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(
            f"No completed Phase 2 {variant} run found for {modelname}."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant", choices=("clean", "pooled", "control"), required=True
    )
    parser.add_argument("--source-run", type=Path)
    parser.add_argument(
        "--source-epoch",
        type=int,
        help="Checkpoint epoch to continue from; defaults to the source run's final epoch.",
    )
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
    corpora, cfg = build_continuation(
        args.variant, source_run, args.device, source_epoch=args.source_epoch
    )
    print_plan(f"Kreuze Phase 2b: {args.variant}", corpora, cfg)
    if args.dry_run:
        return

    main_train(corpora, cfg)
    config_path = Path(cfg.run_dir) / "run_config.json"
    config = json.loads(config_path.read_text())
    config["experiment"] = f"kreuze-phase2b-{args.variant}"
    config["source_run"] = str(source_run)
    config["source_epoch"] = cfg.direction_epoch_offset
    config_path.write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    main()
