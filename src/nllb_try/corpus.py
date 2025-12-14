import os
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import config
from .downloadtatoeba import main_download
from .csv_list_loader import load_parallel_table, find_variety_files

class BaseParallelCorpus:
    """Generic parallel corpus with a standardized interface."""

    def __init__(self, source_lang_nllb, target_lang_nllb):
        self.source_lang_nllb = source_lang_nllb
        self.target_lang_nllb = target_lang_nllb

        self.df = self.load()
        self.df_train, self.df_validate = self.split(self.df)

    def load(self):
        raise NotImplementedError("Subclass must implement load().")

    def split(self, df: pd.DataFrame):
        return train_test_split(df, test_size=0.02, random_state=9358)


class TatoebaCorpus(BaseParallelCorpus):
    """Parallel corpus created from Tatoeba. (typically iso-639-3 format, e.g. gos)"""

    def __init__(self, sl_tat, tl_tat, sl_nllb, tl_nllb):
        self.sl_tat = sl_tat
        self.tl_tat = tl_tat

        main_download([sl_tat, tl_tat], redownload=False)

        super().__init__(sl_nllb, tl_nllb)

    def load(self):
        return load_tatoeba(self.sl_tat, self.tl_tat)

class VarietyCorpus(BaseParallelCorpus):
    """One CSV/TSV file = one corpus object. Currently Assumes DUTCH as one of the two languages!"""

    def __init__(self, path: str, sl_nllb="nld_Latn", tl_nllb="gos_Latn", sep=";"):
        self.path = path
        self.sep = sep
        super().__init__(sl_nllb, tl_nllb)

    def load(self):
        return load_parallel_table(self.path, sep=self.sep)

def load_tatoeba(src: str, trg: str) -> pd.DataFrame:
    src_file = os.path.join(config["TATOEBA_PATH"], f"{src}_sentences.tsv")
    trg_file = os.path.join(config["TATOEBA_PATH"], f"{trg}_sentences.tsv")
    link_file = os.path.join(config["TATOEBA_PATH"], "links.csv")

    src_df = pd.read_csv(src_file, sep="\t", header=None, names=["id", "lang", "source_sentence"])
    trg_df = pd.read_csv(trg_file, sep="\t", header=None, names=["id", "lang", "target_sentence"])
    link   = pd.read_csv(link_file, sep="\t", header=None, names=["origin", "translation"])

    df = (
        link
        .merge(trg_df, left_on="origin", right_on="id")
        .merge(src_df, left_on="translation", right_on="id")
        [["target_sentence", "source_sentence"]]
    )
    return df

def main_corpus(
    source_langs_tatoeba,
    source_langs_nllb,
    variety_dir=None,
    recursive: bool=True,
    sep: str=";"
):
    """
    Builds:
      • 1 corpus per Tatoeba language pair
      • 1 corpus per variety CSV/TSV file
    """

    corpora = []

    # Tatoeba corpora
    zipped = list(zip(source_langs_tatoeba, source_langs_nllb))

    for i, (sl_tat, sl_nllb) in enumerate(zipped):
        for tl_tat, tl_nllb in zipped[i + 1:]:
            print(f"Setting up Tatoeba corpus for {sl_nllb} - {tl_nllb}")
            corpora.append(
                TatoebaCorpus(sl_tat, tl_tat, sl_nllb, tl_nllb)
            )

    # Variety corpora
    if variety_dir:
        files = find_variety_files(variety_dir, recursive=recursive)
        if not files:
            print(f"No variety files found in: {variety_dir}")
        else:
            for f in files:
                print(f"Loading variety file: {f}")
                corpora.append(VarietyCorpus(f, sep=sep))
    else:
        print("No variety directory provided.")

    return corpora