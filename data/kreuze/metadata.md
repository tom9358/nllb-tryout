# Kreuze synthetic-data metadata

## Source

- Local source copy: `sidetracks/kreuze/sentences_kreuze.json`
- Upstream source: https://github.com/tom9358/kreuze-zuik/blob/main/sentences_kreuze.json
- Source collection: 103 documents and 124,022 raw sentence entries
- Preparation: whitespace normalization, exact source deduplication, and a
  maximum of 48 words and 200 characters per source sentence

The CSV files use the format:

```text
Nederlands;Gronings
```

The JSONL files contain the generated pair plus provenance such as document
and source index.

## Generated corpora

| CSV | Provenance JSONL | Pairs | Generator | Model/checkpoint |
|---|---|---:|---|---|
| `kreuze_synthetic_epoch4.csv` | `kreuze_synthetic_epoch4.jsonl` | 104,441 | `sidetracks/kreuze/generate_kreuze_synthetic.py` | fine-tuned NLLB epoch4 |
| `kreuze_synthetic.csv` | `kreuze_synthetic.jsonl` | 104,441 | `sidetracks/kreuze/generate_kreuze_synthetic.py` | fine-tuned NLLB epoch8 |
| `kreuze_synthetic_epoch12.csv` | `kreuze_synthetic_epoch12.jsonl` | 104,440 | `sidetracks/kreuze/generate_kreuze_synthetic.py` | fine-tuned NLLB epoch12 |
| `kreuze_synthetic_gemma50.csv` | `kreuze_synthetic_gemma50.jsonl` | 104,447 | `sidetracks/kreuze/generate_kreuze_gemma.py` | `hf.co/unsloth/gemma-4-31B-it-GGUF:Q8_K_XL` |

All corpora translate Gronings to standard Dutch. The NLLB runs used local
fine-tuned checkpoints from:

```text
checkpoints/nllb-200-distilled-600M-nld-gos-deu-eng-spa-20260823-185018/checkpoints/
```

The Gemma corpus was generated from 2026-08-28 to 2026-08-29 with
temperature `0`, seed `9358`, ten fixed Tatoeba training examples in one user
prompt, and thinking disabled. It requested 50 consecutive sentences per
query and fell back to 25, 5, or 1 sentence when alignment failed. The run
used 2,268 queries and was resumable.

## Rough quality comparisons

On the same 200 Tatoeba holdout sentences, the best tested Gemma setup was
10-shot prompting in one user message without a system prompt:

- BLEU: about 69
- chrF: about 81
- Exact matches: 101/200

The best tested Qwen setup scored about BLEU 66, chrF 80, and 88/200 exact.
These scores are only rough indicators: valid paraphrases, dialect ambiguity,
and errors in some Tatoeba references are scored too harshly.

Gemma was also tested on randomly grouped blocks of 50 sentences. Eight
blocks (400 sentences) stayed perfectly aligned and scored BLEU 73.34, chrF
83.72, and 193/400 exact. This test used random sentences rather than
coherent passages, so it mainly tested block handling rather than contextual
translation quality.

The NLLB epoch validation scores for Gronings to Dutch were:

| Checkpoint | BLEU | chrF |
|---|---:|---:|
| epoch4 | 71.62 | 82.74 |
| epoch8 | 75.18 | 84.97 |
| epoch12 | 74.84 | 84.29 |

Manual inspection found Gemma outputs substantially more natural than the
NLLB-generated alternatives, especially when translating coherent passages.
Gemma still makes lexical and idiomatic mistakes, so the corpus is synthetic
training data and not gold reference data.

## Intended use

The Gemma corpus is the preferred candidate for future synthetic-data
experiments. The NLLB corpora are retained as comparison data for now. The
Gradio comparison viewer is:

```bash
uv run sidetracks/kreuze/inspect_kreuze.py
```
