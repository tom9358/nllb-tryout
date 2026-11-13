"""Experiment tracking on the Hugging Face Hub (opt-in).

When ``tracking_project`` is set in ``RunConfig``:

* training loss and evaluation scores are logged locally with Trackio
  (https://huggingface.co/docs/trackio) and, when ``trackio_space_id`` is set,
  synced to that (public, static) Space at the end of every train/eval step;
* when ``splits_repo_id`` is also set, the exact data of every run is uploaded
  to that private dataset repo, under ``runs/<run name>/train/`` (the corpora
  the model was trained on) and ``runs/<run name>/eval/`` (the validation
  splits it was evaluated on).
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

from .artifacts import write_json

_active: dict | None = None


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _corpus_name(i: int, corpus) -> str:
    path = getattr(corpus, "path", None)
    label = Path(path).stem if path else "tatoeba"
    return f"corpus{i}_{label}_{corpus.source_lang_nllb}-{corpus.target_lang_nllb}"


def _upload_splits(corpus_objects, repo_id, path_in_repo, splits) -> str:
    """Upload the given splits as parquet files and return the dataset revision."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        rows = {}
        for i, corpus in enumerate(corpus_objects):
            for split in splits:
                name = f"{_corpus_name(i, corpus)}_{split}"
                df = getattr(corpus, f"df_{split}")
                df.to_parquet(Path(tmp) / f"{name}.parquet", index=False)
                rows[name] = len(df)
        write_json(Path(tmp) / "manifest.json", {"git_commit": _git_commit(), "rows": rows})
        info = api.upload_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=tmp,
            path_in_repo=path_in_repo,
            commit_message=f"Add {path_in_repo}",
        )
    return info.oid


def start_run(
    run_dir: str,
    train_corpora: list | None = None,
    eval_corpora: list | None = None,
) -> bool:
    """Start or resume the Trackio run of ``run_dir``; False if tracking is off.

    Uploaded data revisions are added to ``run_config.json``.
    """
    global _active

    config_path = Path(run_dir) / "run_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not config.get("tracking_project"):
        return False
    import trackio

    run_name = Path(run_dir).name
    repo_id = config.get("splits_repo_id")
    if repo_id and train_corpora is not None:
        config["train_data_revision"] = _upload_splits(
            train_corpora, repo_id, f"runs/{run_name}/train", ("train", "validate")
        )
    if repo_id and eval_corpora is not None:
        config["eval_data_revision"] = _upload_splits(
            eval_corpora, repo_id, f"runs/{run_name}/eval", ("validate",)
        )
    config.setdefault("git_commit", _git_commit())
    write_json(config_path, config)

    trackio.init(
        project=config["tracking_project"],
        name=run_name,
        config=config,
        resume="allow",
    )
    _active = {"project": config["tracking_project"], "space_id": config.get("trackio_space_id")}
    return True


def log_train(step: int, epoch: int, loss: float) -> None:
    if _active:
        import trackio

        trackio.log({"train/loss": loss, "train/epoch": epoch}, step=step)


def log_eval(df_results: pd.DataFrame) -> None:
    """Log one entry per evaluated checkpoint (``eval/epoch`` 0 is the baseline)."""
    if not _active:
        return
    import trackio

    for _, row in df_results.iterrows():
        name = str(row["Training Steps"])
        metrics = {
            f"eval/{k}": float(v)
            for k, v in row.items()
            if k != "Training Steps" and pd.notna(v)
        }
        metrics["eval/epoch"] = 0 if name == "baseline" else int(name.removeprefix("epoch"))
        trackio.log(metrics)


def finish() -> None:
    global _active
    if _active:
        import trackio

        trackio.finish()
        if _active["space_id"]:
            # Live (Gradio) Spaces need HF PRO; static Spaces are free but public.
            trackio.sync(project=_active["project"], space_id=_active["space_id"], sdk="static")
        _active = None
