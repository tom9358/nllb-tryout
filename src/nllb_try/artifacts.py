from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable


def init_run_dir(run_dir: str) -> dict[str, Path]:
    """Create the standard run directory layout and return key paths."""
    run_path = Path(run_dir)
    checkpoints_dir = run_path / "checkpoints"
    train_dir = run_path / "train"

    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    train_dir.mkdir(parents=True, exist_ok=True)

    return {
        "run_dir": run_path,
        "checkpoints_dir": checkpoints_dir,
        "train_dir": train_dir,
    }


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_text(path: str | Path, text: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def format_run_config_txt(run_config: dict[str, Any]) -> str:
    """Human-readable summary derived from run_config.json."""
    lines: list[str] = []
    for k in sorted(run_config.keys()):
        lines.append(f"{k}: {run_config[k]}")
    return "\n".join(lines) + "\n"


def write_loss_csv(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    rows_list = list(rows)
    if not rows_list:
        return

    fieldnames = list(rows_list[0].keys())
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_list)
