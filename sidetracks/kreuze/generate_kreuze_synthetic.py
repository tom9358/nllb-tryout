"""Create synthetic Gronings-Dutch pairs from the Kreuze sentence collection.

The output TSV is intentionally compatible with ``VarietyCorpus``:

    Nederlands<TAB>Gronings

Usage:
    .venv/bin/python sidetracks/kreuze/generate_kreuze_synthetic.py \
        --model-path checkpoints/<run>/checkpoints/epoch12 \
        --output data/kreuze/kreuze_synthetic.tsv

Use ``--limit`` first to inspect a small pilot batch.  A JSONL sidecar is
written next to the TSV with source-document and model provenance.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import urllib.request
from pathlib import Path

import torch
from tqdm.auto import tqdm

from nllb_try.evaluate import translate
from nllb_try.tokenizer_and_model_setup import setup_model_and_tokenizer

DEFAULT_SOURCE = Path(__file__).with_name("sentences_kreuze.json")
REPO_ROOT = Path(__file__).resolve().parents[2]
WHITESPACE_RE = re.compile(r"\s+")


def load_sentences(source: str | Path) -> list[dict[str, object]]:
    """Load, normalize, and deduplicate source sentences."""
    source_text = str(source)
    if source_text.startswith(("http://", "https://")):
        with urllib.request.urlopen(source_text) as response:
            documents = json.load(response)
    else:
        with Path(source).open(encoding="utf-8") as source_file:
            documents = json.load(source_file)

    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for document, sentences in documents.items():
        for index, raw_sentence in enumerate(sentences):
            sentence = WHITESPACE_RE.sub(" ", raw_sentence).strip()
            if not sentence or sentence in seen:
                continue
            seen.add(sentence)
            rows.append(
                {
                    "gronings": sentence,
                    "document": document,
                    "source_index": index,
                }
            )
    return rows


def sort_by_token_length(
    rows: list[dict[str, object]], tokenizer, max_input_length: int
) -> list[dict[str, object]]:
    """Group similarly sized inputs to reduce padding within each batch."""
    lengths: list[int] = []
    for start in range(0, len(rows), 1024):
        batch = rows[start : start + 1024]
        encoded = tokenizer(
            [str(row["gronings"]) for row in batch],
            truncation=True,
            max_length=max_input_length,
            padding=False,
        )
        lengths.extend(len(input_ids) for input_ids in encoded["input_ids"])
    return [
        row
        for _, row in sorted(zip(lengths, rows), key=lambda item: item[0], reverse=True)
    ]


def write_pair(
    output_writer,
    metadata_file,
    gronings: str,
    dutch: str,
    source: dict[str, object],
    model_path: str,
    source_url: str,
) -> None:
    output_writer.writerow([dutch, gronings])
    metadata_file.write(
        json.dumps(
            {
                "dutch": dutch,
                "gronings": gronings,
                "document": source["document"],
                "source_index": source["source_index"],
                "model_path": model_path,
                "source_url": source_url,
            },
            ensure_ascii=False,
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="Checkpoint directory.")
    parser.add_argument(
        "--source",
        "--source-url",
        dest="source",
        default=str(DEFAULT_SOURCE),
        help=f"Local Kreuze JSON file or URL (default: {DEFAULT_SOURCE})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/kreuze/kreuze_synthetic.tsv"),
        help="Output TSV path (default: data/kreuze/kreuze_synthetic.tsv).",
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        help="JSONL provenance path (default: output path with .jsonl suffix).",
    )
    parser.add_argument("--device", default="cuda", help="PyTorch device.")
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Translation batch size."
    )
    parser.add_argument(
        "--max-input-length",
        type=int,
        default=200,
        help="Maximum tokenizer input length.",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=48,
        help="Skip source entries longer than this many whitespace-delimited words.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=200,
        help="Skip source entries longer than this many characters.",
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=4,
        help="Beam count for deterministic generation.",
    )
    parser.add_argument(
        "--separator",
        default=";",
        help="Output delimiter (default: ';', matching the training loader).",
    )
    parser.add_argument(
        "--sort-by-length",
        action="store_true",
        help="Group sentences by tokenizer length to reduce batch padding.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Only translate the first N entries, useful for a pilot run.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing output files instead of failing.",
    )
    args = parser.parse_args()

    if Path.cwd().resolve() != REPO_ROOT:
        parser.error(f"Run this script from the repository root: {REPO_ROOT}")
    if args.batch_size < 1 or args.max_input_length < 1 or args.num_beams < 1:
        parser.error(
            "--batch-size, --max-input-length, and --num-beams must be positive"
        )
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error(f"CUDA device requested but CUDA is unavailable: {args.device}")

    metadata_output = args.metadata_output or args.output.with_suffix(".jsonl")
    if not args.overwrite:
        existing = [path for path in (args.output, metadata_output) if path.exists()]
        if existing:
            parser.error(
                "Output already exists; use --overwrite or choose another path: "
                + ", ".join(map(str, existing))
            )

    rows = [
        row
        for row in load_sentences(args.source)
        if len(str(row["gronings"]).split()) <= args.max_words
        and len(str(row["gronings"])) <= args.max_chars
    ]
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be positive")
        rows = rows[: args.limit]
    if not rows:
        parser.error("No source sentences remain after filtering.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Translating {len(rows):,} Gronings sentences with {args.model_path}")
    model, tokenizer = setup_model_and_tokenizer(
        args.model_path,
        new_lang="gos_Latn",
        device=args.device,
    )
    if args.sort_by_length:
        rows = sort_by_token_length(rows, tokenizer, args.max_input_length)
        print("Sorted inputs by tokenizer length to reduce batch padding.")

    written = 0
    skipped = 0
    with (
        args.output.open("w", encoding="utf-8", newline="") as output_file,
        metadata_output.open("w", encoding="utf-8") as metadata_file,
    ):
        output_writer = csv.writer(
            output_file, delimiter=args.separator, lineterminator="\n"
        )
        output_writer.writerow(["Nederlands", "Gronings"])
        for start in tqdm(range(0, len(rows), args.batch_size), unit="batch"):
            batch = rows[start : start + args.batch_size]
            gronings = [str(row["gronings"]) for row in batch]
            dutch_batch = translate(
                gronings,
                src_lang="gos_Latn",
                tgt_lang="nld_Latn",
                model=model,
                tokenizer=tokenizer,
                max_input_length=args.max_input_length,
                num_beams=args.num_beams,
            )
            if len(dutch_batch) != len(batch):
                raise RuntimeError(
                    f"Model returned {len(dutch_batch)} translations for "
                    f"{len(batch)} inputs."
                )

            for source, dutch in zip(batch, dutch_batch):
                dutch = WHITESPACE_RE.sub(" ", dutch).strip()
                if not dutch:
                    skipped += 1
                    continue
                write_pair(
                    output_writer,
                    metadata_file,
                    str(source["gronings"]),
                    dutch,
                    source,
                    args.model_path,
                    args.source,
                )
                written += 1
            output_file.flush()
            metadata_file.flush()

    print(f"Wrote {written:,} pairs to {args.output}; skipped {skipped:,}.")
    print(f"Wrote provenance to {metadata_output}.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
