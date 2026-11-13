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
import re
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
    _active = {
        "project": config["tracking_project"],
        "space_id": config.get("trackio_space_id"),
        "config_path": config_path,
        "epoch_end_steps": {},
    }
    return True


_CARD_START = "<!-- training-runs:start -->"
_CARD_END = "<!-- training-runs:end -->"


def _commit_url(sha: str | None) -> str | None:
    if not sha:
        return None
    try:
        remote = subprocess.check_output(
            ["git", "remote", "get-url", "origin"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    remote = remote.removesuffix(".git").replace("git@github.com:", "https://github.com/")
    return f"{remote}/commit/{sha}" if remote.startswith("https://github.com/") else None


def _card_section(repo_id: str, run_dirs: list[str]) -> str:
    lines = [
        _CARD_START,
        "## Training data and runs",
        "",
        "This section is generated on upload. Each training stage below links to",
        "the exact data it was trained and evaluated on (pinned to a dataset",
        "revision), its full configuration, and the code version that ran it.",
        "",
    ]
    private_data = False
    for run_dir in run_dirs:
        name = Path(run_dir).name
        cfg = json.loads((Path(run_dir) / "run_config.json").read_text(encoding="utf-8"))
        lines.append(f"### `{name}`")
        lines.append("")
        lines.append(
            f"- Configuration: [`training/{name}/run_config.json`]"
            f"(https://huggingface.co/{repo_id}/blob/main/training/{name}/run_config.json)"
        )
        data_repo = cfg.get("splits_repo_id")
        for kind, label in (("train", "Training data"), ("eval", "Evaluation data")):
            rev = cfg.get(f"{kind}_data_revision")
            if data_repo and rev:
                private_data = True
                lines.append(
                    f"- {label}: [`{data_repo}` @ `{rev[:7]}`, `runs/{name}/{kind}/`]"
                    f"(https://huggingface.co/datasets/{data_repo}/tree/{rev}/runs/{name}/{kind})"
                )
        if cfg.get("trackio_space_id"):
            lines.append(
                f"- Loss and scores: Trackio project `{cfg['tracking_project']}`, run `{name}`"
                f" on https://huggingface.co/spaces/{cfg['trackio_space_id']}"
            )
        sha = cfg.get("git_commit")
        if sha:
            url = _commit_url(sha)
            lines.append(f"- Code: [`{sha[:7]}`]({url})" if url else f"- Code: commit `{sha[:7]}`")
        lines.append("")
    if private_data:
        lines.append("The data repo is private; ask the model author for access.")
        lines.append("")
    lines.append(_CARD_END)
    return "\n".join(lines)


def publish_run_info(repo_id: str, run_dirs: list[str]) -> None:
    """Upload run configs to ``training/<run name>/`` of the model repo and
    (re)write the "Training data and runs" section of its model card."""
    from huggingface_hub import CommitOperationAdd, HfApi
    from huggingface_hub.utils import EntryNotFoundError

    api = HfApi()
    try:
        card = Path(api.hf_hub_download(repo_id, "README.md")).read_text(encoding="utf-8")
    except EntryNotFoundError:
        card = ""
    section = _card_section(repo_id, run_dirs)
    if _CARD_START in card and _CARD_END in card:
        head, rest = card.split(_CARD_START, 1)
        card = head + section + rest.split(_CARD_END, 1)[1]
    else:
        card = card.rstrip() + "\n\n" + section + "\n"

    ops = [CommitOperationAdd("README.md", card.encode("utf-8"))]
    for run_dir in run_dirs:
        for fname in ("run_config.json", "training_data_manifest.json"):
            path = Path(run_dir) / fname
            if path.exists():
                ops.append(CommitOperationAdd(f"training/{Path(run_dir).name}/{fname}", str(path)))
    api.create_commit(repo_id, operations=ops, commit_message="Add training run info")


def log_train(step: int, epoch: int, loss: float) -> None:
    if _active:
        import trackio

        trackio.log({"train/loss": loss}, step=step)
        _active["epoch_end_steps"][str(epoch)] = step


_SCORE_COLUMN = re.compile(
    r"^(corpus\d+)_(train|validate)_(bleu|chrf)_(\S+?)_to_(\S+?)_(?:src_to_tgt|tgt_to_src)$"
)


def _metric_names(columns) -> dict[str, str]:
    """Map score columns to readable names such as ``chrF (validation)/nld→gos``."""
    names = {}
    for col in columns:
        m = _SCORE_COLUMN.match(col)
        if not m:
            continue
        corpus, split, metric, src, tgt = m.groups()
        group = f"{'BLEU' if metric == 'bleu' else 'chrF'} ({'train' if split == 'train' else 'validation'})"
        name = f"{group}/{src.split('_')[0]}→{tgt.split('_')[0]}"
        names[col] = (name, corpus)
    counts = {}
    for name, _ in names.values():
        counts[name] = counts.get(name, 0) + 1
    # Same language pair in several corpora: add the corpus to tell them apart.
    return {
        col: name if counts[name] == 1 else f"{name} ({corpus})"
        for col, (name, corpus) in names.items()
    }


def log_eval(df_results: pd.DataFrame) -> None:
    """Log the scores of every evaluated checkpoint at the training step where
    that checkpoint was saved (the baseline at step 0)."""
    if not _active:
        return
    import trackio

    config = json.loads(_active["config_path"].read_text(encoding="utf-8"))
    epoch_end_steps = config.get("epoch_end_steps", {})
    names = _metric_names(df_results.columns)
    for _, row in df_results.iterrows():
        name = str(row["Training Steps"])
        step = 0 if name == "baseline" else epoch_end_steps.get(name.removeprefix("epoch"))
        metrics = {names[k]: float(v) for k, v in row.items() if k in names and pd.notna(v)}
        trackio.log(metrics, step=step)


def finish() -> None:
    global _active
    if _active:
        import trackio

        trackio.finish()
        if _active["epoch_end_steps"]:
            config = json.loads(_active["config_path"].read_text(encoding="utf-8"))
            config["epoch_end_steps"] = _active["epoch_end_steps"]
            write_json(_active["config_path"], config)
        if _active["space_id"]:
            # Live (Gradio) Spaces need HF PRO; static Spaces are free but public.
            trackio.sync(project=_active["project"], space_id=_active["space_id"], sdk="static")
        _active = None
