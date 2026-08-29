# LLM investigation for synthetic Kreuze data

## Motivation

I had written roughly 7,000 Gronings sentences by hand for Tatoeba. The
fine-tuned NLLB model was improving, but not quickly enough to make serious
progress without much more data. The idea was to use the existing model, or a
larger local LLM, to backtranslate a much larger Gronings text collection into
Dutch.

Kreuze was chosen as the source because its texts are freely available online
and explicitly state:

> “Vrij rond te sturen noar elkenain dij t lezen wil.”

The source and attribution information are retained with the generated data.

## 1. NLLB backtranslation

The first attempt used the fine-tuned NLLB model itself. Epochs 4, 8, and 12
were tested because they had similar Gronings-to-Dutch validation scores,
while training-set scores kept increasing:

| Checkpoint | Validation BLEU | Validation chrF |
|---|---:|---:|
| epoch4 | 71.62 | 82.74 |
| epoch8 | 75.18 | 84.97 |
| epoch12 | 74.84 | 84.29 |

Heuristic checks looked for untranslated Gronings words and manual inspection
looked at names, lexical choices, phrasing, and obvious nonsense. The outputs
were immediately disappointing. They contained many wrong word choices,
mistranslated names, unnatural sentences, and Gronings remnants. The three
NLLB corpora were kept as comparison data, but were not considered valuable
synthetic training data.

The NLLB generation was fast, taking roughly 15 minutes per full corpus. This
would have been attractive if the quality had been sufficient.

## 2. Qwen experiments

The local `Qwen3.6-35B-A3B-GGUF:Q8_K_XL` model was tested through the local
OpenAI-compatible server. The first small probe was mixed: some translations
were good, but it made errors with pronouns, names, and dialect words.

A 20-sentence Tatoeba test with a small Gronings glossary produced:

- BLEU: 75.94
- chrF: 86.26

Few-shot prompting was then tested on a fixed 200-sentence Tatoeba holdout.
The few-shot examples came from the training split and were excluded from
the test set.

| Prompt setup | BLEU | chrF | Exact |
|---|---:|---:|---:|
| 0-shot with system prompt | 59.33 | 75.03 | 64/200 |
| 5-shot, one user prompt, with system | 64.34 | 79.18 | 81/200 |
| 10-shot, one user prompt, with system | 64.16 | 79.56 | 81/200 |
| 5-shot, one user prompt, no system | 64.21 | 79.05 | 80/200 |
| **10-shot, one user prompt, no system** | **66.40** | **80.04** | **88/200** |
| 10-shot with extra “natural, idiomatic” wording | 66.05 | 79.73 | 88/200 |
| 20-shot with the same extra wording | 64.76 | 79.55 | 79/200 |
| 10-shot, temperature 0.3 | 64.81 | 78.96 | 87/200 |

The best Qwen setup used one user message, ten examples, no system prompt,
and temperature 0. The extra wording and additional examples did not help.

## 3. Gemma experiments

The local `gemma-4-31B-it-GGUF:Q8_K_XL` model was then tested under the same
general conditions. In zero-shot mode it often returned explanations and
alternative translations instead of only the translation. Few-shot examples
caused it to follow the desired output format much more reliably.

The following results use the same 200-sentence Tatoeba holdout:

| Prompt setup | BLEU | chrF | Exact |
|---|---:|---:|---:|
| 0-shot, no system | 55.85 | 73.31 | 64/200 |
| 0-shot, with system | 56.80 | 72.96 | 65/200 |
| 5-shot, no system | 68.60 | 80.12 | 96/200 |
| 5-shot, with system | 69.07 | 80.39 | 95/200 |
| **10-shot, no system** | **68.97** | **80.58** | **101/200** |
| 10-shot, with system | 68.84 | 80.57 | 99/200 |

Gemma was better than Qwen on this test, especially in exact matches. Manual
inspection also found more natural Dutch than in the NLLB outputs.

### Qwen and Gemma side by side

The following table combines the prompt experiments on the same
200-sentence Tatoeba holdout. The broad setup was comparable, but the exact
prompt wording was not perfectly identical between the two model runs, so
these results should be treated as a strong indication rather than a
strictly controlled head-to-head benchmark.

| Prompt setup | Qwen BLEU | Qwen chrF | Qwen exact | Gemma BLEU | Gemma chrF | Gemma exact |
|---|---:|---:|---:|---:|---:|---:|
| 0-shot, with system | 59.33 | 75.03 | 64/200 | 56.80 | 72.96 | 65/200 |
| 5-shot, with system | 64.34 | 79.18 | 81/200 | 69.07 | 80.39 | 95/200 |
| 5-shot, no system | 64.21 | 79.05 | 80/200 | 68.60 | 80.12 | 96/200 |
| 10-shot, with system | 64.16 | 79.56 | 81/200 | 68.84 | 80.57 | 99/200 |
| **10-shot, no system** | **66.40** | **80.04** | **88/200** | **68.97** | **80.58** | **101/200** |

Gemma outperformed Qwen in these few-shot runs, with the largest difference
in exact matches. Both models benefited substantially from few-shot
prompting, but adding more than ten examples was not tested as a likely
high-value direction for the final bulk-generation setup.

Some Gemma errors were genuine lexical or idiomatic errors, for example
misinterpreting words such as `brogge`, `schieterg`, `ootje`, `liepen`, and
`n tik`. Other apparent mismatches were not necessarily errors:

- `transparant` was correct in one case even though the Tatoeba reference
  incorrectly used “doorzichtig”;
- Gronings `ie` and `joe` can mean either “u” or “jullie”;
- natural paraphrases were penalized by exact match and BLEU.

The scores therefore underestimate actual quality in some cases.

## 4. Longer context blocks

Gemma was tested on consecutive Kreuze sentences rather than isolated
sentences. The prompt asked for exactly one Dutch line per input sentence, in
the original order.

- Five-sentence blocks: 5/5 blocks returned the correct number of lines.
- Ten-sentence blocks: 3/3 blocks returned the correct number of lines.
- Twenty-sentence blocks: 3/3 blocks returned the correct number of lines.
- Fifty-sentence blocks: 2/2 blocks returned the correct number of lines.

Context helped with references and ambiguous words. In a passage about water
plants, for example, `kreuze` was translated as “kroos” when the surrounding
sentences supplied the context. Context also improved some interpretations in
the fishing and story passages.

As a quantitative batching test, eight blocks of 50 randomly grouped Tatoeba
sentences were translated in eight queries:

- 400/400 output lines were aligned;
- no fallback was needed;
- BLEU: 73.34;
- chrF: 83.72;
- exact matches: 193/400.

Those sentences were randomly grouped, so this primarily tested block
handling rather than real discourse context. Coherent Kreuze passages may
benefit more.

## 5. Full Gemma generation

The full Kreuze collection was generated with Gemma in blocks of 50
consecutive sentences. The fallback order was:

```text
50 → 25 → 5 → 1
```

The generator checked the exact number of output lines after every request and
flushed CSV and JSONL after every completed block. It supports resuming an
interrupted run.

The resulting corpus contains 104,447 Gronings–Dutch pairs from 103 source
documents. It required 2,268 API queries. The effective row counts were:

| Effective block size | Rows |
|---:|---:|
| 50 | 101,800 |
| 25 | 2,125 |
| 5 | 470 |
| 1 | 5 |
| final block of 47 | 47 |

The output files are:

```text
data/kreuze/kreuze_synthetic_gemma50.csv
data/kreuze/kreuze_synthetic_gemma50.jsonl
```

## Final chosen setup

The current preferred synthetic-data setup is:

- model: `hf.co/unsloth/gemma-4-31B-it-GGUF:Q8_K_XL`;
- Gronings → standard Dutch;
- one user prompt, no system prompt;
- ten fixed Gronings–Dutch examples from the Tatoeba training data;
- temperature `0`;
- seed `9358`;
- thinking disabled;
- 50 consecutive Kreuze sentences per request;
- strict line-count validation;
- fallback to 25, 5, or 1 sentence;
- resumable CSV and JSONL output with provenance.

The local source copy used by the generators is
`sidetracks/kreuze/sentences_kreuze.json`. The Gemma corpus is currently the
preferred candidate for further synthetic-data and NLLB-training
experiments. It remains synthetic data and should not be treated as gold
reference data without further quality filtering.

The follow-up NLLB experiments, validation safeguards and staged-training
ideas are specified in [`training_plan.md`](training_plan.md).
