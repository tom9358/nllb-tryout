import csv

import pytest

from generate_kreuze_phase1_results_table import (
    END_MARKER,
    METRICS,
    START_MARKER,
    Experiment,
    generate_table,
    read_result,
    replace_generated_table,
)


def write_metrics(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["Training Steps", *(column for _, column in METRICS)]
    with path.open("w", newline="", encoding="utf-8") as metrics_file:
        writer = csv.DictWriter(metrics_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_generate_table_reads_selected_row(tmp_path):
    metrics_path = tmp_path / "metrics.csv"
    values = {column: str(index + 0.125) for index, (_, column) in enumerate(METRICS)}
    write_metrics(
        metrics_path,
        [
            {"Training Steps": "epoch1", **values},
            {"Training Steps": "epoch2", **values},
        ],
    )
    experiment = Experiment(
        "600M", "Synthetic pooled base", 902, "metrics.csv", "epoch2"
    )

    table = generate_table(tmp_path, (experiment,))

    assert "| 600M | Synthetic pooled base | 902 |" in table
    assert "| 0.12 | 1.12 | 2.12 | 3.12 | 4.12 | 5.12 | 6.12 | 7.12 |" in table


def test_read_result_rejects_missing_row(tmp_path):
    metrics_path = tmp_path / "metrics.csv"
    write_metrics(metrics_path, [{"Training Steps": "epoch1"}])
    experiment = Experiment("600M", "Missing", 0, "metrics.csv", "epoch2")

    with pytest.raises(ValueError, match="found 0"):
        read_result(tmp_path, experiment)


def test_replace_generated_table():
    document = f"Before\n{START_MARKER}\nold\n{END_MARKER}\nAfter\n"

    updated = replace_generated_table(document, "new")

    assert updated == f"Before\n{START_MARKER}\nnew\n{END_MARKER}\nAfter\n"
