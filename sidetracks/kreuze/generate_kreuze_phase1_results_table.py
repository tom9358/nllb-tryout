"""Generate the complete two-language Kreuze experiment results table."""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

START_MARKER = "<!-- BEGIN GENERATED PHASE 1 RESULTS -->"
END_MARKER = "<!-- END GENERATED PHASE 1 RESULTS -->"


@dataclass(frozen=True)
class Experiment:
    model_size: str
    stage: str
    steps: int
    metrics_path: str
    row_name: str


EXPERIMENTS = (
    Experiment(
        "600M",
        "Untrained baseline",
        0,
        "checkpoints/nllb-200-distilled-600M-nld-gos-kreuze-phase1-control-20260829-193625/eval/kreuze-phase1-full-validation/metrics.csv",
        "baseline",
    ),
    Experiment(
        "600M",
        "Tatoeba-only base, epoch 1",
        451,
        "checkpoints/nllb-200-distilled-600M-nld-gos-kreuze-phase1-control-20260829-193625/eval/kreuze-phase1-full-validation/metrics.csv",
        "epoch1",
    ),
    Experiment(
        "600M",
        "Synthetic pooled base, epoch 1",
        451,
        "checkpoints/nllb-200-distilled-600M-nld-gos-kreuze-phase1-pooled-20260829-193625/eval/kreuze-phase1-full-validation/metrics.csv",
        "epoch1",
    ),
    Experiment(
        "600M",
        "Tatoeba-only base, epoch 2",
        902,
        "checkpoints/nllb-200-distilled-600M-nld-gos-kreuze-phase1-control-20260829-193625/eval/kreuze-phase1-full-validation/metrics.csv",
        "epoch2",
    ),
    Experiment(
        "600M",
        "Synthetic pooled base, epoch 2",
        902,
        "checkpoints/nllb-200-distilled-600M-nld-gos-kreuze-phase1-pooled-20260829-193625/eval/kreuze-phase1-full-validation/metrics.csv",
        "epoch2",
    ),
    Experiment(
        "600M",
        "Tatoeba-only extended control",
        954,
        "checkpoints/nllb-200-distilled-600M-nld-gos-kreuze-phase1b-control-20260829-200440/eval/kreuze-phase1-full-validation/metrics.csv",
        "epoch1",
    ),
    Experiment(
        "600M",
        "Synthetic + pooled continuation",
        954,
        "checkpoints/nllb-200-distilled-600M-nld-gos-kreuze-phase1b-pooled-20260829-195150/eval/kreuze-phase1-full-validation/metrics.csv",
        "epoch1",
    ),
    Experiment(
        "600M",
        "Synthetic + clean finish",
        954,
        "checkpoints/nllb-200-distilled-600M-nld-gos-kreuze-phase1b-clean-20260829-195150/eval/kreuze-phase1-full-validation/metrics.csv",
        "epoch1",
    ),
    Experiment(
        "1.3B",
        "Untrained baseline",
        0,
        "checkpoints/nllb-200-distilled-1.3B-nld-gos-kreuze-phase1-control-20260829-203443/eval/kreuze-phase1-full-validation/metrics.csv",
        "baseline",
    ),
    Experiment(
        "1.3B",
        "Tatoeba-only base, epoch 1",
        451,
        "checkpoints/nllb-200-distilled-1.3B-nld-gos-kreuze-phase1-control-20260829-203443/eval/kreuze-phase1-full-validation/metrics.csv",
        "epoch1",
    ),
    Experiment(
        "1.3B",
        "Synthetic pooled base, epoch 1",
        451,
        "checkpoints/nllb-200-distilled-1.3B-nld-gos-kreuze-phase1-pooled-20260829-203443/eval/kreuze-phase1-full-validation/metrics.csv",
        "epoch1",
    ),
    Experiment(
        "1.3B",
        "Tatoeba-only base, epoch 2",
        902,
        "checkpoints/nllb-200-distilled-1.3B-nld-gos-kreuze-phase1-control-20260829-203443/eval/kreuze-phase1-full-validation/metrics.csv",
        "epoch2",
    ),
    Experiment(
        "1.3B",
        "Synthetic pooled base, epoch 2",
        902,
        "checkpoints/nllb-200-distilled-1.3B-nld-gos-kreuze-phase1-pooled-20260829-203443/eval/kreuze-phase1-full-validation/metrics.csv",
        "epoch2",
    ),
    Experiment(
        "1.3B",
        "Tatoeba-only extended control",
        954,
        "checkpoints/nllb-200-distilled-1.3B-nld-gos-kreuze-phase1b-control-20260829-204754/eval/kreuze-phase1-full-validation/metrics.csv",
        "epoch1",
    ),
    Experiment(
        "1.3B",
        "Synthetic + pooled continuation",
        954,
        "checkpoints/nllb-200-distilled-1.3B-nld-gos-kreuze-phase1b-pooled-20260829-204957/eval/kreuze-phase1-full-validation/metrics.csv",
        "epoch1",
    ),
    Experiment(
        "1.3B",
        "Synthetic + clean finish",
        954,
        "checkpoints/nllb-200-distilled-1.3B-nld-gos-kreuze-phase1b-clean-20260829-204754/eval/kreuze-phase1-full-validation/metrics.csv",
        "epoch1",
    ),
)

METRICS = (
    ("Tatoeba NL→GOS BLEU", "corpus0_validate_bleu_nld_Latn_to_gos_Latn_src_to_tgt"),
    ("Tatoeba NL→GOS chrF", "corpus0_validate_chrf_nld_Latn_to_gos_Latn_src_to_tgt"),
    ("Tatoeba GOS→NL BLEU", "corpus0_validate_bleu_gos_Latn_to_nld_Latn_tgt_to_src"),
    ("Tatoeba GOS→NL chrF", "corpus0_validate_chrf_gos_Latn_to_nld_Latn_tgt_to_src"),
    ("Kreuze NL→GOS BLEU", "corpus1_validate_bleu_nld_Latn_to_gos_Latn_src_to_tgt"),
    ("Kreuze NL→GOS chrF", "corpus1_validate_chrf_nld_Latn_to_gos_Latn_src_to_tgt"),
    ("Kreuze GOS→NL BLEU", "corpus1_validate_bleu_gos_Latn_to_nld_Latn_tgt_to_src"),
    ("Kreuze GOS→NL chrF", "corpus1_validate_chrf_gos_Latn_to_nld_Latn_tgt_to_src"),
)


def read_result(repo_root: Path, experiment: Experiment) -> dict[str, str]:
    metrics_path = repo_root / experiment.metrics_path
    with metrics_path.open(newline="", encoding="utf-8") as metrics_file:
        rows = list(csv.DictReader(metrics_file))

    matches = [row for row in rows if row["Training Steps"] == experiment.row_name]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one {experiment.row_name!r} row in {metrics_path}, "
            f"found {len(matches)}"
        )
    return matches[0]


def generate_table(
    repo_root: Path, experiments: tuple[Experiment, ...] = EXPERIMENTS
) -> str:
    headers = ["Model", "Training stage", "Steps", *(label for label, _ in METRICS)]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---", "---", "---:", *(["---:"] * len(METRICS))]) + "|",
    ]

    for experiment in experiments:
        result = read_result(repo_root, experiment)
        values = [
            experiment.model_size,
            experiment.stage,
            f"{experiment.steps:,}",
            *(f"{float(result[column]):.2f}" for _, column in METRICS),
        ]
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def replace_generated_table(document: str, table: str) -> str:
    if document.count(START_MARKER) != 1 or document.count(END_MARKER) != 1:
        raise ValueError("Expected exactly one generated Phase 1 table marker pair")

    before, remainder = document.split(START_MARKER, maxsplit=1)
    _, after = remainder.split(END_MARKER, maxsplit=1)
    return f"{before}{START_MARKER}\n{table}\n{END_MARKER}{after}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help="Replace the generated table in data/kreuze/training_plan.md.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    table = generate_table(repo_root)
    if not args.write:
        print(table)
        return

    document_path = repo_root / "data/kreuze/training_plan.md"
    document = document_path.read_text(encoding="utf-8")
    document_path.write_text(
        replace_generated_table(document, table),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
