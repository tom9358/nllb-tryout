# NLLB Gronings Tatoeba Translation Machine

This project focuses on building a translation machine for the Gronings language using the Tatoeba dataset and the NLLB (No Language Left Behind) model.

I'm working on this in my free time so the project does not always have steady progress.

A first version of the translation model is now on huggingface! https://huggingface.co/Tom9358/nllb-tatoeba-gos-nld-v1

Here is a Huggingface space where the model can be used: https://huggingface.co/spaces/Tom9358/gos_gronings_translate

(Here is also an equivalent google colab: https://colab.research.google.com/drive/1b5dn3VT4fvOBKly1CIx4Qwo59GDM1H-M)

## Training

It was trained on about 10.000 Gronings-Dutch sentence pairs from [Tatoeba](https://tatoeba.org/), about half of which I wrote myself.

I tried my best to check for naturalness and spelling using the Gronings online dictionary and corpus [Woordwaark](https://woordwaark.nl/), and the Gronings-language website [dideldom.nu](https://dideldom.nu/). Particularly the [Kreuze](https://dideldom.nu/kreuze) Gronings magazines hosted there I found very useful, and I wrote a little [search interface](https://tom9358.pythonanywhere.com/) to easily find example sentences in those magazines. I never copied any sentences and instead always formulated analogous ones myself.

### Additional parallel data

Training can include one or more parallel-data files or directories:

```bash
uv run run_train.py \
  --preset pooled \
  --parallel-data-path data/kreuze/kreuze_synthetic_gemma50.csv
```

`--parallel-data-path` may be repeated. Directories are searched recursively
for `.csv` and `.tsv` files. Each file must:

- be UTF-8 encoded (a UTF-8 byte-order mark is accepted);
- contain a header and exactly two columns;
- use NLLB language labels as headers, with source first and target second,
  for example `nld_Latn;gos_Latn`;
- contain non-empty values in both columns.

Blank pairs are removed, but sentence text is otherwise left unchanged.
By default, `.csv` uses `;` and `.tsv` uses a tab. Use
`--parallel-data-separator` to override the delimiter for all configured
parallel files.

## Thanks

A heartfelt thanks to the authors in Kreuze, to the team behind Woordwaark, and to the hoster of dideldom! Without you, I would have been nowhere.

Special thanks to [CmdCody](https://huggingface.co/CmdCody/) for the very similar and very inspirational project for North Frisian, and for the link to a useful blogpost.

Thanks to the nice blogpost [How to Fine-Tune a NLLB-200 Model for Translating a New Language](https://cointegrated.medium.com/how-to-fine-tune-a-nllb-200-model-for-translating-a-new-language-a37fc706b865) for helping me get started and helping with some parts of the code.

Thanks to Tatoeba for including Gronings as one of the languages on their site, for letting me add and correct sentences there in many languages (I've written hundreds of English, German and Spanish translation equivalents of Gronings sentences as well!), and for letting me download this data as a parallel corpus dataset.
