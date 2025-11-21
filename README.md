# NLLB Gronings Tatoeba Translation Machine

This project focuses on building a translation machine for the Gronings language using the Tatoeba dataset and the NLLB (No Language Left Behind) model.

I'm working on this in my free time so the project does not always have steady progress.

A first version of the translation model is now on huggingface! https://huggingface.co/Tom9358/nllb-tatoeba-gos-nld-v1

Here is a google colab notebook where the model can be used: https://colab.research.google.com/drive/1b5dn3VT4fvOBKly1CIx4Qwo59GDM1H-M

## Training

It was trained on about 10.000 Gronings-Dutch sentence pairs from [Tatoeba](https://tatoeba.org/), about half of which I wrote myself.

I tried my best to check for naturalness and spelling using the Gronings online dictionary and corpus [Woordwaark](https://woordwaark.nl/), and the Gronings-language website [dideldom.nu](https://dideldom.nu/). Particularly the [Kreuze](https://dideldom.nu/kreuze) Gronings magazines hosted there I found very useful, and I wrote a little [search interface](https://tom9358.pythonanywhere.com/) to easily find example sentences in those magazines. I never copied any sentences and instead always formulated analogous ones myself.

## Thanks

A heartfelt thanks to the authors in Kreuze, to the team behind Woordwaark, and to the hoster of dideldom! Without you, I would have been nowhere.

Special thanks to [CmdCody](https://huggingface.co/CmdCody/) for the very similar and very inspirational project for North Frisian, and for the link to a useful blogpost.

Thanks to the nice blogpost [How to Fine-Tune a NLLB-200 Model for Translating a New Language](https://cointegrated.medium.com/how-to-fine-tune-a-nllb-200-model-for-translating-a-new-language-a37fc706b865) for helping me get started and helping with some parts of the code.

Thanks to Tatoeba for including Gronings as one of the languages on their site, for letting me add and correct sentences there in many languages (I've written hundreds of English, German and Spanish translation equivalents of Gronings sentences as well!), and for letting me download this data as a parallel corpus dataset.
