"""Compare generated Kreuze corpora in a local Gradio web UI.

Usage:
    .venv/bin/python sidetracks/kreuze/inspect_kreuze.py
"""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

from audit_kreuze_synthetic import load_metadata, load_rows

DATA_DIR = Path("data/kreuze")
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATHS = {
    "epoch4": (
        DATA_DIR / "kreuze_synthetic_epoch4.csv",
        DATA_DIR / "kreuze_synthetic_epoch4.jsonl",
    ),
    "epoch8": (
        DATA_DIR / "kreuze_synthetic_epoch8.csv",
        DATA_DIR / "kreuze_synthetic_epoch8.jsonl",
    ),
    "epoch12": (
        DATA_DIR / "kreuze_synthetic_epoch12.csv",
        DATA_DIR / "kreuze_synthetic_epoch12.jsonl",
    ),
    "gemma50": (
        DATA_DIR / "kreuze_synthetic_gemma50.csv",
        DATA_DIR / "kreuze_synthetic_gemma50.jsonl",
    ),
}
PAGE_SIZE = 50
WHITESPACE_RE = re.compile(r"\s+")
MODEL_LABELS = {
    "epoch4": "Epoch4 Nederlands",
    "epoch8": "Epoch8 Nederlands",
    "epoch12": "Epoch12 Nederlands",
    "gemma50": "Gemma Nederlands",
}
DEFAULT_VISIBLE_MODELS = ["epoch8", "gemma50"]


def normalize(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip().casefold()


def load_dataset(
    corpus_path: Path, metadata_path: Path, separator: str
) -> dict[str, dict[str, object]]:
    rows = load_rows(corpus_path, separator)
    metadata = load_metadata(metadata_path)
    if len(rows) != len(metadata):
        raise ValueError(
            f"{corpus_path}: corpus rows ({len(rows)}) and metadata rows "
            f"({len(metadata)}) differ."
        )
    return {
        row["Gronings"]: {"dutch": row["Nederlands"], "metadata": source}
        for row, source in zip(rows, metadata)
    }


def build_records(
    datasets: dict[str, dict[str, dict[str, object]]],
) -> list[dict[str, object]]:
    common_sources = set.intersection(*(set(dataset) for dataset in datasets.values()))
    reference = datasets["epoch8"]
    return [
        {
            "gronings": source,
            **{name: datasets[name][source]["dutch"] for name in datasets},
            "metadata": reference[source]["metadata"],
        }
        for source in reference
        if source in common_sources
    ]


def table_config(
    indices: list[int],
    records: list[dict[str, object]],
    visible_models: list[str],
) -> tuple[list[str], list[str], list[list[object]]]:
    visible_models = [
        model for model in DEFAULT_PATHS if model in (visible_models or [])
    ]
    fields = ["gronings", *visible_models]
    headers = ["#", "Gronings", *[MODEL_LABELS[model] for model in visible_models]]
    datatypes = ["number", *["str"] * len(fields)]
    datatypes.extend(["str", "number"])
    headers.extend(["Document", "Source index"])
    return headers, datatypes, table_rows(indices, records, visible_models)


def table_rows(
    indices: list[int],
    records: list[dict[str, object]],
    visible_models: list[str],
) -> list[list[object]]:
    result = []
    for index in indices:
        record = records[index]
        source = record["metadata"]
        result.append(
            [
                index + 1,
                record["gronings"],
                *[record[model] for model in visible_models],
                source.get("document", ""),
                source.get("source_index", ""),
            ]
        )
    return result


def page_view(
    indices: list[int],
    page: int,
    records: list[dict[str, object]],
    visible_models: list[str],
) -> tuple[list[list[object]], str, int]:
    page_count = max(1, (len(indices) + PAGE_SIZE - 1) // PAGE_SIZE)
    page = max(1, min(int(page or 1), page_count))
    start = (page - 1) * PAGE_SIZE
    end = min(start + PAGE_SIZE, len(indices))
    return (
        table_rows(indices[start:end], records, visible_models),
        f"Showing {start + 1:,}-{end:,} of {len(indices):,} matching rows "
        + f"(page {page}/{page_count})",
        page,
    )


def build_app(records: list[dict[str, object]]):
    import gradio as gr

    all_indices = list(range(len(records)))

    def render(indices: list[int], page: int, visible_models: list[str]):
        headers, datatypes, _ = table_config(indices, records, visible_models)
        table, status, page = page_view(indices, page, records, visible_models)
        return (
            gr.update(
                headers=headers,
                datatype=datatypes,
                value=table,
            ),
            status,
            indices,
            page,
        )

    def search(query: str, visible_models: list[str]):
        query = normalize(query or "")
        if query:
            indices = [
                index
                for index, record in enumerate(records)
                if any(
                    query in normalize(str(record[field]))
                    for field in ("gronings", *DEFAULT_PATHS)
                )
            ]
        else:
            indices = all_indices
        return render(indices, 1, visible_models)

    def random_rows(visible_models: list[str]):
        indices = random.sample(all_indices, min(PAGE_SIZE, len(all_indices)))
        table, status, indices, page = render(indices, 1, visible_models)
        return table, f"Random sample: {status}", indices, page

    def change_page(indices: list[int], page: int, visible_models: list[str]):
        rendered = render(indices or all_indices, page, visible_models)
        return rendered[0], rendered[1], rendered[3]

    initial_headers, initial_datatypes, initial_table = table_config(
        all_indices, records, DEFAULT_VISIBLE_MODELS
    )
    _, initial_status, _ = page_view(all_indices, 1, records, DEFAULT_VISIBLE_MODELS)

    with gr.Blocks(title="Kreuze synthetic corpus") as app:
        gr.Markdown(
            "# Compare Kreuze synthetic corpora\n"
            "Search Dutch or Gronings text and compare epoch4, epoch8, epoch12, "
            "and Gemma translations side by side."
        )
        matches = gr.State(all_indices)
        with gr.Row():
            query = gr.Textbox(
                label="Search",
                placeholder="Search either language...",
                scale=4,
            )
            search_button = gr.Button("Search", variant="primary", scale=1)
            random_button = gr.Button("Random sample", scale=1)
        visible_models = gr.CheckboxGroup(
            choices=[(label, model) for model, label in MODEL_LABELS.items()],
            value=DEFAULT_VISIBLE_MODELS,
            label="Visible translation columns",
            info="Gronings and provenance stay visible.",
        )
        with gr.Row():
            previous_button = gr.Button("Previous")
            page = gr.Number(label="Page", value=1, precision=0, minimum=1)
            next_button = gr.Button("Next")
        status = gr.Markdown(initial_status)
        table = gr.Dataframe(
            headers=[
                *initial_headers,
            ],
            datatype=initial_datatypes,
            value=initial_table,
            interactive=False,
            wrap=True,
            max_height=700,
        )

        search_button.click(
            search,
            inputs=[query, visible_models],
            outputs=[table, status, matches, page],
        )
        query.submit(
            search,
            inputs=[query, visible_models],
            outputs=[table, status, matches, page],
        )
        random_button.click(
            random_rows,
            inputs=visible_models,
            outputs=[table, status, matches, page],
        )
        previous_button.click(
            lambda indices, current_page, models: change_page(
                indices, current_page - 1, models
            ),
            inputs=[matches, page, visible_models],
            outputs=[table, status, page],
        )
        next_button.click(
            lambda indices, current_page, models: change_page(
                indices, current_page + 1, models
            ),
            inputs=[matches, page, visible_models],
            outputs=[table, status, page],
        )
        page.change(
            change_page,
            inputs=[matches, page, visible_models],
            outputs=[table, status, page],
        )
        visible_models.change(
            lambda query_text, models: search(query_text, models),
            inputs=[query, visible_models],
            outputs=[table, status, matches, page],
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--separator", default=";")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    for epoch, (corpus, metadata) in DEFAULT_PATHS.items():
        parser.add_argument(f"--{epoch}-corpus", type=Path, default=corpus)
        parser.add_argument(f"--{epoch}-metadata", type=Path, default=metadata)
    args = parser.parse_args()

    if Path.cwd().resolve() != REPO_ROOT:
        raise SystemExit(f"Run this script from the repository root: {REPO_ROOT}")

    datasets = {
        epoch: load_dataset(
            getattr(args, f"{epoch}_corpus"),
            getattr(args, f"{epoch}_metadata"),
            args.separator,
        )
        for epoch in DEFAULT_PATHS
    }
    records = build_records(datasets)
    if not records:
        raise ValueError("The corpora have no common Gronings sources.")
    print(f"Loaded {len(records):,} common sources across all corpora.")

    app = build_app(records)
    app.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
