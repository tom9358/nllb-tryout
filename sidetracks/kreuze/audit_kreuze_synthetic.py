"""Audit a generated Kreuze synthetic corpus.

The default checks are lightweight and do not load a model:

    .venv/bin/python sidetracks/kreuze/audit_kreuze_synthetic.py

Round-trip scoring is optional and translates a sample of Dutch outputs back
to Gronings with a checkpoint:

    .venv/bin/python sidetracks/kreuze/audit_kreuze_synthetic.py \
        --model-path checkpoints/<run>/checkpoints/epoch12 \
        --roundtrip-sample 512
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

DEFAULT_CORPUS = Path("data/kreuze/kreuze_synthetic.csv")
DEFAULT_METADATA = Path("data/kreuze/kreuze_synthetic.jsonl")
WHITESPACE_RE = re.compile(r"\s+")
TOKEN_RE = re.compile(r"[A-Za-zÀ-ÿ]+(?:['’][A-Za-zÀ-ÿ]+)?")
GRONINGS_MARKERS = (
    "nait",
    "bie",
    "joar",
    "hai",
    "zien",
    "mit",
    "doar",
    "oet",
    "noar",
    "dou",
    "wuir",
    "vot",
)


def normalize(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip().casefold()


def describe(values: list[int]) -> str:
    ordered = sorted(values)
    return (
        f"median={statistics.median(values):g}, "
        f"p95={ordered[int(0.95 * len(values))]}, max={max(values)}"
    )


def load_rows(path: Path, separator: str) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file, delimiter=separator)
        expected = {"Nederlands", "Gronings"}
        if set(reader.fieldnames or ()) != expected:
            raise ValueError(
                f"{path}: expected columns {sorted(expected)}, "
                f"found {reader.fieldnames}"
            )
        rows = list(reader)
    if any(set(row) != expected for row in rows):
        raise ValueError(f"{path}: found rows with unexpected columns")
    return rows


def load_metadata(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def load_tatoeba_gos(path: Path) -> set[str]:
    sentences: set[str] = set()
    with path.open(encoding="utf-8", errors="replace") as input_file:
        for line in input_file:
            fields = line.rstrip("\n").split("\t", 2)
            if len(fields) == 3:
                sentences.add(normalize(fields[2]))
    return sentences


def print_examples(
    rows: list[dict[str, str]], scores: list[float], count: int
) -> None:
    print("\nWorst round-trip examples:")
    for index in sorted(range(len(rows)), key=lambda i: scores[i])[:count]:
        row = rows[index]
        print(
            f"CHRF={scores[index]:.1f} | "
            f"GOS={row['Gronings']} | NL={row['Nederlands']}"
        )


def roundtrip_audit(
    rows: list[dict[str, str]],
    model_path: str,
    sample_size: int,
    batch_size: int,
    device: str,
    num_beams: int,
    seed: int,
    worst_examples: int,
) -> None:
    import pandas as pd
    from sacrebleu.metrics import CHRF

    from src.nllb_try.evaluate import translate
    from src.nllb_try.tokenizer_and_model_setup import setup_model_and_tokenizer

    sample_size = min(sample_size, len(rows))
    indices = random.Random(seed).sample(range(len(rows)), sample_size)
    sampled = [rows[index] for index in indices]
    model, tokenizer = setup_model_and_tokenizer(
        model_path, new_lang="gos_Latn", device=device
    )

    translations: list[str] = []
    for start in range(0, sample_size, batch_size):
        batch = sampled[start : start + batch_size]
        translations.extend(
            translate(
                [row["Nederlands"] for row in batch],
                src_lang="nld_Latn",
                tgt_lang="gos_Latn",
                model=model,
                tokenizer=tokenizer,
                max_input_length=200,
                num_beams=num_beams,
            )
        )

    metric = CHRF()
    scores = [
        metric.sentence_score(back, [row["Gronings"]]).score
        for row, back in zip(sampled, translations)
    ]
    print(
        f"\nRound-trip CHRF ({sample_size:,} sampled pairs): "
        f"mean={statistics.mean(scores):.2f}, "
        f"median={statistics.median(scores):.2f}, "
        f"p10={sorted(scores)[int(0.10 * len(scores))]:.2f}, "
        f"p90={sorted(scores)[int(0.90 * len(scores))]:.2f}"
    )
    print("Round-trip scores below:", ", ".join(
        f"{threshold}={sum(score < threshold for score in scores):,}"
        for threshold in (20, 30, 40, 50, 60, 70)
    ))
    if worst_examples:
        print_examples(sampled, scores, min(worst_examples, len(scores)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--separator", default=";")
    parser.add_argument("--tatoeba-gos", type=Path, default=Path("data/tatoeba/gos_sentences.tsv"))
    parser.add_argument("--max-words", type=int, default=48)
    parser.add_argument("--max-chars", type=int, default=200)
    parser.add_argument("--model-path", help="Enable optional round-trip scoring.")
    parser.add_argument("--roundtrip-sample", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--seed", type=int, default=9358)
    parser.add_argument("--worst-examples", type=int, default=15)
    args = parser.parse_args()

    if args.max_words < 1 or args.max_chars < 1:
        parser.error("--max-words and --max-chars must be positive")
    if args.model_path and (args.roundtrip_sample < 1 or args.batch_size < 1):
        parser.error("round-trip sample and batch size must be positive")

    failures: list[str] = []
    try:
        rows = load_rows(args.corpus, args.separator)
        metadata = load_metadata(args.metadata)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    sources = [row["Gronings"] for row in rows]
    targets = [row["Nederlands"] for row in rows]
    source_keys = [normalize(text) for text in sources]
    target_keys = [normalize(text) for text in targets]
    source_words = [len(text.split()) for text in sources]
    target_words = [len(text.split()) for text in targets]
    source_chars = [len(text) for text in sources]
    target_chars = [len(text) for text in targets]

    malformed = [
        index + 2
        for index, row in enumerate(rows)
        if not row["Gronings"].strip() or not row["Nederlands"].strip()
    ]
    if malformed:
        failures.append(f"{len(malformed):,} rows have empty fields")

    if len(rows) != len(metadata):
        failures.append(
            f"corpus has {len(rows):,} rows but metadata has {len(metadata):,}"
        )

    print(f"Corpus: {args.corpus}")
    print(f"Rows: {len(rows):,}")
    print(
        f"Unique Gronings: {len(set(source_keys)):,} "
        f"(duplicates: {len(rows) - len(set(source_keys)):,})"
    )
    print(
        f"Unique Dutch: {len(set(target_keys)):,} "
        f"(duplicates: {len(rows) - len(set(target_keys)):,})"
    )
    print(f"Source words: {describe(source_words)}")
    print(f"Target words: {describe(target_words)}")
    print(f"Source characters: {describe(source_chars)}")
    print(f"Target characters: {describe(target_chars)}")
    print(
        f"Targets over limits: words>{args.max_words}="
        f"{sum(value > args.max_words for value in target_words):,}, "
        f"chars>{args.max_chars}="
        f"{sum(value > args.max_chars for value in target_chars):,}"
    )

    control_chars = sum(
        any(ord(char) < 32 and char not in "\t\n\r" for char in text)
        for text in sources + targets
    )
    print(f"Unexpected control-character fields: {control_chars:,}")
    if control_chars:
        failures.append(f"{control_chars:,} fields contain control characters")

    marker_rows = [
        row
        for row in rows
        if any(
            re.search(rf"\b{re.escape(marker)}\b", row["Nederlands"].casefold())
            for marker in GRONINGS_MARKERS
        )
    ]
    print(
        f"Dutch outputs containing Gronings-like markers: {len(marker_rows):,} "
        f"({100 * len(marker_rows) / len(rows):.2f}%)"
    )

    if args.tatoeba_gos.exists():
        tatoeba_sources = load_tatoeba_gos(args.tatoeba_gos)
        overlap = sum(source in tatoeba_sources for source in source_keys)
        print(f"Exact source overlap with Tatoeba Gronings: {overlap:,}")
    else:
        print(f"Tatoeba file not found; overlap check skipped: {args.tatoeba_gos}")

    if args.model_path:
        roundtrip_audit(
            rows,
            args.model_path,
            args.roundtrip_sample,
            args.batch_size,
            args.device,
            args.num_beams,
            args.seed,
            args.worst_examples,
        )

    if failures:
        print("\nFAILURES:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("\nAudit completed without structural failures.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
