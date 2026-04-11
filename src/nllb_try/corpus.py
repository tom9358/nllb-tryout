import os
import random

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import RunConfig, get_default_config
from .downloadtatoeba import main_download
from .csv_list_loader import load_parallel_table, find_variety_files

# ---------------------------------------------------------------------------
# Validation-split strategy
# ---------------------------------------------------------------------------
# Tatoeba sentences have unique integer IDs, and the same sentence can appear
# in many language pairs (e.g. a Dutch sentence may be linked to English,
# German, AND Gronings translations). If we split each pair independently,
# a sentence can land in the *validation* set of one pair while being in the
# *training* set of another, as the model has seen the exact encoder/decoder
# input during training, which inflates validation scores.
#
# To prevent this we perform a **global sentence-ID hold-out**: we collect
# every unique Tatoeba sentence ID across all pairs, randomly set aside 2%
# of them, and assign any translation pair that touches a held-out ID to
# validation in *every* corpus. Because IDs are shared across pairs, the
# effective validation percentage per pair will be higher than 2%.
# In one brief test this was ~3-4%, which I find completely acceptable.
#
# VarietyCorpus (dialect CSV files) has no Tatoeba IDs and keeps its own
# independent random split.
# ---------------------------------------------------------------------------

GLOBAL_HOLDOUT_FRACTION = 0.02
SPLIT_SEED = 9358

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
        return train_test_split(df, test_size=0.02, random_state=SPLIT_SEED)


class TatoebaCorpus(BaseParallelCorpus):
    """Parallel corpus created from Tatoeba.

    Unlike VarietyCorpus, splitting is **deferred** – the constructor loads
    the data (including Tatoeba sentence IDs) but does NOT split.  Splitting
    happens later in ``main_corpus`` via ``_global_tatoeba_split`` so that
    a single, globally consistent set of held-out sentence IDs is used across
    all Tatoeba language pairs.
    """

    def __init__(self, sl_tat, tl_tat, sl_nllb, tl_nllb, cfg: RunConfig):
        self.sl_tat = sl_tat
        self.tl_tat = tl_tat
        self.cfg = cfg

        main_download([sl_tat, tl_tat], redownload=False, tatoeba_path=cfg.tatoeba_path)

        # Intentionally skip super().__init__ to avoid the automatic split.
        self.source_lang_nllb = sl_nllb
        self.target_lang_nllb = tl_nllb
        self.df = self.load()
        # df_train / df_validate are set later by _global_tatoeba_split()
        self.df_train: pd.DataFrame | None = None
        self.df_validate: pd.DataFrame | None = None

    def load(self):
        return load_tatoeba(self.sl_tat, self.tl_tat, cfg=self.cfg)

class VarietyCorpus(BaseParallelCorpus):
    """One CSV/TSV file = one corpus object. Currently Assumes DUTCH as one of the two languages!"""

    def __init__(self, path: str, sl_nllb="nld_Latn", tl_nllb="gos_Latn", sep=";"):
        self.path = path
        self.sep = sep
        super().__init__(sl_nllb, tl_nllb)

    def load(self):
        return load_parallel_table(self.path, sep=self.sep)

def load_tatoeba(src: str, trg: str, cfg: RunConfig) -> pd.DataFrame:
    """Load a Tatoeba language pair, keeping sentence IDs for global splitting.

    Returns a DataFrame with columns:
        src_id, trg_id, source_sentence, target_sentence
    The ID columns are used by ``_global_tatoeba_split`` and are dropped
    before the data reaches the training loop.
    """
    src_file = os.path.join(cfg.tatoeba_path, f"{src}_sentences.tsv")
    trg_file = os.path.join(cfg.tatoeba_path, f"{trg}_sentences.tsv")
    link_file = os.path.join(cfg.tatoeba_path, "links.csv")

    src_df = pd.read_csv(src_file, sep="\t", header=None, names=["id", "lang", "source_sentence"])
    trg_df = pd.read_csv(trg_file, sep="\t", header=None, names=["id", "lang", "target_sentence"])
    link   = pd.read_csv(link_file, sep="\t", header=None, names=["origin", "translation"])

    df = (
        link
        .merge(trg_df, left_on="origin", right_on="id")
        .merge(src_df, left_on="translation", right_on="id")
    )
    df = df.rename(columns={"id_x": "trg_id", "id_y": "src_id"})
    return df[["src_id", "trg_id", "source_sentence", "target_sentence"]]

def _global_tatoeba_split(tatoeba_corpora: list[TatoebaCorpus]) -> None:
    """Split all Tatoeba corpora using a single global set of held-out sentence IDs.

    1. Collect every unique sentence ID across all pairs.
    2. Randomly hold out ``GLOBAL_HOLDOUT_FRACTION`` of them (seeded for
       reproducibility).
    3. For each corpus, any translation pair where *either* the source or
       target ID is in the held-out set goes to validation; the rest goes
       to training.
    4. Drop the ID columns so downstream code sees only
       ``source_sentence`` / ``target_sentence``.

    Because a single sentence ID can appear in several language pairs, the
    effective validation fraction per pair can be higher than
    ``GLOBAL_HOLDOUT_FRACTION`` (the effect seems limited, ~3-4 %).
    """
    # Step 1 — collect all sentence IDs
    all_ids: set[int] = set()
    for corpus in tatoeba_corpora:
        all_ids.update(corpus.df["src_id"])
        all_ids.update(corpus.df["trg_id"])

    # Step 2 — sample held-out IDs
    all_ids_sorted = sorted(all_ids)
    holdout_n = int(len(all_ids_sorted) * GLOBAL_HOLDOUT_FRACTION)
    rng = random.Random(SPLIT_SEED)
    holdout_ids = set(rng.sample(all_ids_sorted, holdout_n))

    print(f"Global Tatoeba split: {len(all_ids_sorted):,} unique sentence IDs, "
          f"{holdout_n:,} ({GLOBAL_HOLDOUT_FRACTION:.0%}) held out for validation")

    # Step 3 — split each corpus
    sentence_cols = ["source_sentence", "target_sentence"]
    for corpus in tatoeba_corpora:
        is_val = (
            corpus.df["src_id"].isin(holdout_ids)
            | corpus.df["trg_id"].isin(holdout_ids)
        )
        corpus.df_validate = corpus.df.loc[is_val, sentence_cols].reset_index(drop=True)
        corpus.df_train    = corpus.df.loc[~is_val, sentence_cols].reset_index(drop=True)

        pair = f"{corpus.source_lang_nllb}-{corpus.target_lang_nllb}"
        total = len(corpus.df)
        n_val = len(corpus.df_validate)
        print(f"  {pair}: {total:,} total -> {n_val:,} val ({100*n_val/total:.1f}%), "
              f"{total - n_val:,} train")

        # Step 4 — drop IDs from df too (no longer needed)
        corpus.df = corpus.df[sentence_cols].reset_index(drop=True)


def main_corpus(
    source_langs_tatoeba,
    source_langs_nllb,
    variety_dir=None,
    recursive: bool=True,
    sep: str=";",
    cfg: RunConfig | None = None,
):
    """
    Builds:
      • 1 corpus per Tatoeba language pair
      • 1 corpus per variety CSV/TSV file

    Tatoeba corpora are split with a **global sentence-ID hold-out** so that
    no sentence appearing in any validation set can also appear in any
    training set (across all language pairs).  See module docstring for
    rationale.
    """

    cfg = cfg or get_default_config()

    # Tatoeba corpora (split is deferred)
    tatoeba_corpora: list[TatoebaCorpus] = []
    zipped = list(zip(source_langs_tatoeba, source_langs_nllb))

    for i, (sl_tat, sl_nllb) in enumerate(zipped):
        for tl_tat, tl_nllb in zipped[i + 1:]:
            print(f"Setting up Tatoeba corpus for {sl_nllb} - {tl_nllb}")
            tatoeba_corpora.append(
                TatoebaCorpus(sl_tat, tl_tat, sl_nllb, tl_nllb, cfg=cfg)
            )

    # Perform the global split across all Tatoeba corpora at once
    _global_tatoeba_split(tatoeba_corpora)

    corpora: list[BaseParallelCorpus] = list(tatoeba_corpora)

    # Variety corpora (independent split, no Tatoeba IDs)
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
