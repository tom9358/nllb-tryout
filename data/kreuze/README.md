# Kreuze synthetic-data tools

These sidetrack utilities prepare and inspect synthetic Gronings–Dutch data
from the [Kreuze sentence collection](https://github.com/tom9358/kreuze-zuik/blob/main/sentences_kreuze.json).
The source copy used by the generators is versioned locally in
`sidetracks/kreuze/sentences_kreuze.json`.

Run the commands below from the repository root:

```text
<repository-root>/
```

## Tools

- `generate_kreuze_synthetic.py` generates pairs with a local fine-tuned NLLB checkpoint.
- `generate_kreuze_gemma.py` generates pairs with the local Gemma API in 50-sentence blocks, with fallbacks to 25, 5, and 1 sentence.
- `audit_kreuze_synthetic.py` checks corpus structure and optionally performs round-trip diagnostics.
- `inspect_kreuze.py` provides a Gradio viewer for comparing NLLB epoch4, epoch8, epoch12, and Gemma outputs.

Generated CSV and JSONL files are stored under `data/kreuze/`.
All corpus metadata and conclusions are documented in
`data/kreuze/metadata.md`.

## NLLB generation

```bash
uv run sidetracks/kreuze/generate_kreuze_synthetic.py \
  --model-path checkpoints/<run>/checkpoints/epoch8 \
  --output data/kreuze/kreuze_synthetic.csv
```

## Gemma generation

The Gemma generator saves after every completed block. Resume an interrupted
run with:

```bash
uv run sidetracks/kreuze/generate_kreuze_gemma.py \
  --output data/kreuze/kreuze_synthetic_gemma50.csv \
  --metadata-output data/kreuze/kreuze_synthetic_gemma50.jsonl \
  --resume
```

The API endpoint is read from `LLAMA_API_ENDPOINT` in `.env`.

## Auditing and inspection

```bash
uv run sidetracks/kreuze/audit_kreuze_synthetic.py
uv run sidetracks/kreuze/inspect_kreuze.py
```

The viewer searches the shared Gronings sources and lets you toggle the
translation columns on and off to compare the four generated datasets.
