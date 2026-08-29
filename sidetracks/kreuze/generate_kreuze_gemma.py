"""Generate synthetic Dutch translations with the local Gemma API.

The default mode translates 50 consecutive Kreuze sentences per request and
falls back to 25, 5, and finally 1 sentence if the response is not aligned.

Usage:
    uv run sidetracks/kreuze/generate_kreuze_gemma.py \
        --output data/kreuze/kreuze_synthetic_gemma50.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DEFAULT_SOURCE = Path(__file__).with_name("sentences_kreuze.json")
DEFAULT_MODEL = "hf.co/unsloth/gemma-4-31B-it-GGUF:Q8_K_XL"
DEFAULT_ENDPOINT = os.getenv("LLAMA_API_ENDPOINT")
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = Path("data/kreuze/kreuze_synthetic_gemma50.csv")
DEFAULT_METADATA = Path("data/kreuze/kreuze_synthetic_gemma50.jsonl")
WHITESPACE_RE = re.compile(r"\s+")
NUMBERED_LINE_RE = re.compile(r"^\s*\d+\s*[.)]\s*(.+)$")

FEW_SHOT_EXAMPLES = (
    (
        "t Bliekt dat dit n hail gunstege faktor is veur ons projekt, dat juust as doul het om ien soamenwaarken n netwaark te scheppen van vertoalens ien zoveul meugelk toalen.",
        "Het blijkt dat dit een zeer gunstige factor is voor ons project, dat juist als doel heeft in samenwerking een netwerk te scheppen van vertalingen in zoveel mogelijk talen.",
    ),
    (
        "Je mouten nait vergeten dat, om wat van ain cultuur over te droagen noar n aander, de eerste kondietsie is woorden te bruken dij begrepen worden zellen.",
        "Men mag niet vergeten, dat, om iets uit één cultuur naar een andere over te brengen, de eerste voorwaarde is, woorden te gebruiken, die zullen begrepen worden.",
    ),
    (
        "t Is aaltied meugelk om n onsjuch stel lu te verainegen in laifde, zolaank der aander lu overblieven om heur agressieve utings te inkasseren.",
        "Het is altijd mogelijk om een aanzienlijk aantal mensen te verenigen in liefde, zolang er andere mensen overblijven om hun agressieve uitingen te incasseren.",
    ),
    (
        "Wie kinnen deur tied raaizen. En wie dudden dat mit n hoast nait te gleuven tempo van ain sekonde per sekonde.",
        "We kunnen reizen door de tijd. En we doen dat met de ongelooflijke snelheid van een seconde per seconde.",
    ),
    (
        "Der binnen drij verschillende soorten mìnsen op wereld: dijent dij tellen kinnen en dijen dij dat nait kinnen.",
        "Er zijn drie verschillende soorten mensen op de wereld: zij die kunnen tellen en zij die dat niet kunnen.",
    ),
    (
        "Ale femilieleden van mien vraauw binnen mien schoonfemilie en dus is heur bruier mien swoager, heur zuster mien schoonzuster, mien bruier en zuster binnen heur swoager en schoonzuster.",
        "Alle verwanten van mijn vrouw zijn mijn schoonfamilie, en dus is haar broer mijn schoonbroer, haar zus is mijn schoonzus, mijn broer en zus zijn haar schoonbroer en schoonzus.",
    ),
    (
        "De experts vruigen aan tien dailnemers n riege van vief siefers te onholden dik t aine noa t aandere in t midden van n schaarm verschenen.",
        "De experts vroegen aan tien deelnemers een rij van vijf getallen te onthouden die het ene na het andere in het midden van een scherm verschenen.",
    ),
    (
        "t Is mor tien groaden, en hai lopt in n t-shirt boeten. k Krieg t al kold as k noar hom kiek.",
        "Het is maar tien graden, en hij loopt in een T-shirt buiten. Ik krijg het al koud als ik naar hem kijk.",
    ),
    (
        "Dokters dochten dat e dood was, mor vandoag is e gezond en wel en hai het waark en n femilie.",
        "De dokters dachten dat hij dood was, maar vandaag is hij gezond en wel en hij heeft werk en een familie.",
    ),
    (
        "Dou k mien ogen weer opendee, ston der ien ainen n onbekende doame veur mien neus.",
        "Toen ik mijn ogen weer opendeed, stond er ineens een onbekende dame voor mijn neus.",
    ),
)


def load_sentences(source: str | Path) -> list[dict[str, object]]:
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
        for source_index, raw_sentence in enumerate(sentences):
            sentence = WHITESPACE_RE.sub(" ", raw_sentence).strip()
            if not sentence or sentence in seen:
                continue
            seen.add(sentence)
            if len(sentence.split()) > 48 or len(sentence) > 200:
                continue
            rows.append(
                {
                    "gronings": sentence,
                    "document": document,
                    "source_index": source_index,
                }
            )
    return rows


def parse_output(output: str, expected: int) -> list[str] | None:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if len(lines) != expected:
        return None

    cleaned: list[str] = []
    for line in lines:
        match = NUMBERED_LINE_RE.match(line)
        cleaned.append(match.group(1).strip() if match else line)
    if any(not line or "\t" in line for line in cleaned):
        return None
    return cleaned


class GemmaTranslator:
    def __init__(
        self,
        model: str,
        endpoint: str,
        temperature: float,
        seed: int,
        max_tokens: int,
    ) -> None:
        self.model = model
        self.endpoint = endpoint
        self.temperature = temperature
        self.seed = seed
        self.max_tokens = max_tokens
        self.examples = "\n\n".join(
            f"Gronings: {gronings}\nNederlands: {dutch}"
            for gronings, dutch in FEW_SHOT_EXAMPLES
        )
        self.query_count = 0

    def translate(self, sentences: list[str]) -> str:
        count = len(sentences)
        inputs = "\n".join(
            f"{index}. {sentence}" for index, sentence in enumerate(sentences, 1)
        )
        prompt = (
            f"Here are ten examples of Gronings translated into standard Dutch:\n\n"
            f"{self.examples}\n\n"
            f"Translate the following {count} Gronings sentences into natural Dutch. "
            f"Preserve the number and order of sentences. Return exactly {count} "
            "lines, one Dutch translation per input sentence, with no numbering, "
            f"explanation, labels, or quotation marks.\n\n{inputs}"
        )
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "top_p": 1,
            "seed": self.seed,
            "max_tokens": self.max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        self.query_count += 1
        with urllib.request.urlopen(request, timeout=900) as response:
            result = json.load(response)
        return result["choices"][0]["message"]["content"].strip()


def translate_with_fallback(
    translator: GemmaTranslator,
    rows: list[dict[str, object]],
) -> tuple[list[str], list[int]]:
    count = len(rows)
    output = translator.translate([str(row["gronings"]) for row in rows])
    parsed = parse_output(output, count)
    if parsed is not None:
        return parsed, [count] * count

    if count > 25:
        midpoint = count // 2
        left, left_sizes = translate_with_fallback(translator, rows[:midpoint])
        right, right_sizes = translate_with_fallback(translator, rows[midpoint:])
        return left + right, left_sizes + right_sizes

    if count > 5:
        translations: list[str] = []
        sizes: list[int] = []
        for start in range(0, count, 5):
            result, result_sizes = translate_with_fallback(
                translator, rows[start : start + 5]
            )
            translations.extend(result)
            sizes.extend(result_sizes)
        return translations, sizes

    if count > 1:
        translations = []
        sizes = []
        for row in rows:
            result, result_sizes = translate_with_fallback(translator, [row])
            translations.extend(result)
            sizes.extend(result_sizes)
        return translations, sizes

    raise RuntimeError(
        f"Could not obtain one aligned translation for: {rows[0]['gronings']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        "--source-url",
        dest="source",
        default=str(DEFAULT_SOURCE),
        help=f"Local Kreuze JSON file or URL (default: {DEFAULT_SOURCE})",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--endpoint",
        default=DEFAULT_ENDPOINT,
        help="Chat-completions endpoint (default: LLAMA_API_ENDPOINT from .env).",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata-output", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=9358)
    parser.add_argument("--max-tokens", type=int, default=5000)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing complete output blocks.",
    )
    args = parser.parse_args()

    if Path.cwd().resolve() != REPO_ROOT:
        parser.error(f"Run this script from the repository root: {REPO_ROOT}")
    if not args.endpoint:
        parser.error(
            "No API endpoint configured; set LLAMA_API_ENDPOINT in .env "
            "or pass --endpoint."
        )
    if args.temperature < 0 or args.max_tokens < 1:
        parser.error("--temperature must be non-negative and --max-tokens positive")
    existing = [path for path in (args.output, args.metadata_output) if path.exists()]
    if existing and not (args.overwrite or args.resume):
        parser.error(
            "Output already exists; use --overwrite or choose another path: "
            + ", ".join(map(str, existing))
        )
    if args.overwrite and args.resume:
        parser.error("--overwrite and --resume cannot be combined")

    rows = load_sentences(args.source)
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be positive")
        rows = rows[: args.limit]
    if not rows:
        parser.error("No source sentences remain after filtering.")
    total_rows = len(rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    translator = GemmaTranslator(
        args.model,
        args.endpoint,
        args.temperature,
        args.seed,
        args.max_tokens,
    )
    print(f"Translating {len(rows):,} sentences with {args.model}", flush=True)

    written = 0
    if args.resume:
        if not all(path.exists() for path in (args.output, args.metadata_output)):
            parser.error("--resume requires both output files to exist")
        with args.output.open(encoding="utf-8", newline="") as input_file:
            existing_rows = list(csv.DictReader(input_file, delimiter=";"))
        with args.metadata_output.open(encoding="utf-8") as input_file:
            existing_metadata = [
                json.loads(line) for line in input_file if line.strip()
            ]
        if len(existing_rows) != len(existing_metadata):
            raise RuntimeError(
                "Cannot resume: CSV and JSONL contain different numbers of rows "
                f"({len(existing_rows)} vs {len(existing_metadata)})."
            )
        if len(existing_rows) > len(rows):
            raise RuntimeError(
                f"Cannot resume: existing output has {len(existing_rows):,} rows, "
                f"but source has only {len(rows):,}."
            )
        for index, (output_row, metadata, source) in enumerate(
            zip(existing_rows, existing_metadata, rows)
        ):
            if (
                output_row.get("Gronings") != source["gronings"]
                or metadata.get("gronings") != source["gronings"]
                or metadata.get("dutch") != output_row.get("Nederlands")
            ):
                raise RuntimeError(
                    f"Cannot resume: output mismatch at row {index + 1:,}."
                )
        written = len(existing_rows)
        rows = rows[written:]
        print(
            f"Resuming after {written:,} complete rows; {len(rows):,} remain.",
            flush=True,
        )
    elif args.overwrite:
        written = 0

    fallback_sizes: dict[int, int] = {}
    file_mode = "a" if args.resume else "w"
    with (
        args.output.open(file_mode, encoding="utf-8", newline="") as output_file,
        args.metadata_output.open(file_mode, encoding="utf-8") as metadata_file,
    ):
        writer = csv.writer(output_file, delimiter=";", lineterminator="\n")
        if not args.resume:
            writer.writerow(["nld_Latn", "gos_Latn"])
        for start in range(0, len(rows), 50):
            block = rows[start : start + 50]
            translations, actual_sizes = translate_with_fallback(translator, block)
            if len(translations) != len(block):
                raise RuntimeError(
                    f"Alignment failure at source rows {start}-{start + len(block) - 1}: "
                    f"got {len(translations)} translations for {len(block)} inputs"
                )
            for source, dutch, actual_size in zip(block, translations, actual_sizes):
                dutch = WHITESPACE_RE.sub(" ", dutch).strip()
                if not dutch:
                    raise RuntimeError(
                        f"Empty translation for source: {source['gronings']}"
                    )
                writer.writerow([dutch, source["gronings"]])
                metadata_file.write(
                    json.dumps(
                        {
                            "dutch": dutch,
                            "gronings": source["gronings"],
                            "document": source["document"],
                            "source_index": source["source_index"],
                            "model": args.model,
                            "temperature": args.temperature,
                            "seed": args.seed,
                            "requested_block_size": 50,
                            "actual_block_size": actual_size,
                            "source": args.source,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                fallback_sizes[actual_size] = fallback_sizes.get(actual_size, 0) + 1
                written += 1
            output_file.flush()
            metadata_file.flush()
            print(
                f"Processed {written:,}/{total_rows:,} "
                f"(queries={translator.query_count})",
                flush=True,
            )

    print(f"Wrote {written:,} pairs to {args.output}.")
    print(f"Wrote provenance to {args.metadata_output}.")
    print(f"Effective block sizes: {fallback_sizes}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
