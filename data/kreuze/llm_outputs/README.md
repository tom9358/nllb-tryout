# Saved LLM outputs

This directory contains outputs saved during the Qwen and Gemma experiments
described in `../llm_investigation.md`.

- `gemma10_tatoeba_200.json` contains all 200 Gemma predictions from the
  detailed error-analysis run, together with the Gronings input and Tatoeba
  reference.
- `gemma50_pilot.csv` and `gemma50_pilot.jsonl` contain a small 50-sentence
  Gemma generation pilot and its provenance.
- `qwen_benchmark_outputs.json` contains all Qwen outputs that were printed
  during the original tests. The 20-sentence runs are complete; the
  200-sentence runs contain only the examples printed by the test commands,
  while their aggregate scores cover all 200 sentences.

The Qwen outputs were not originally written to a file. They were recovered
from the session event log where available. No complete 200-output Qwen
prediction list exists.

Prompt templates and the final bulk-generation prompt are documented in
`prompts.md`. Internal API endpoint details are intentionally omitted.
