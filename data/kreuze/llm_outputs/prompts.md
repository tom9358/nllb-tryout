# LLM prompt experiments

All tests used Gronings → standard Dutch. Few-shot examples were taken from
the Tatoeba training split and excluded from the test inputs.

## Qwen prompt variants

The Qwen experiments included:

1. zero-shot with a system instruction;
2. few-shot examples as `user`/`assistant` chat messages;
3. few-shot examples embedded in one user message;
4. one-user-message few-shot prompts with and without a system prompt;
5. 5-, 10-, 15-, and 20-shot variants;
6. temperature `0` and `0.3`.

The best Qwen setup was 10-shot in one user message, without a system
message, at temperature `0`:

```text
Here are ten examples of Gronings translated into standard Dutch:

Gronings: <example>
Nederlands: <translation>

<nine more examples>

Now translate this new sentence. Return only the Dutch translation.
Gronings: <input>
Nederlands:
```

## Gemma prompt variants

Gemma was tested with zero-shot, 5-shot, and 10-shot prompts, with and
without a system message. The best sentence-level setup was 10-shot, one
user message, no system message, and temperature `0`.

The final bulk-generation prompt keeps the ten fixed Tatoeba examples and
adds only the instructions needed to preserve block alignment:

```text
Here are ten examples of Gronings translated into standard Dutch:

Gronings: <example>
Nederlands: <translation>

<nine more examples>

Translate the following <N> Gronings sentences into natural Dutch.
Preserve the number and order of sentences. Return exactly <N> lines,
one Dutch translation per input sentence, with no numbering, explanation,
labels, or quotation marks.

1. <first Gronings sentence>
2. <second Gronings sentence>
...
<N>. <last Gronings sentence>
```

The exact ten fixed examples are stored in
`sidetracks/kreuze/generate_kreuze_gemma.py`.

## Request settings

The relevant generation settings were:

```text
temperature: 0
top_p: 1
thinking: disabled
```

The bulk Gemma generator requested 50 consecutive Kreuze sentences per
request and fell back to 25, 5, or 1 sentence if the output line count did
not match the input count.
