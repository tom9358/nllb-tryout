import os
import random

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import RunConfig, get_default_config
from .csv_list_loader import find_parallel_files, load_parallel_table
from .downloadtatoeba import main_download

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
# ParallelFileCorpus data has no Tatoeba IDs and keeps its own independent
# random split.
# ---------------------------------------------------------------------------

GLOBAL_HOLDOUT_FRACTION = 0.02
SPLIT_SEED = 9358


class BaseParallelCorpus:
    """A parallel corpus: a language pair plus its sentence data.

    Subclasses fetch their data however they like, then hand the result to
    this constructor, which stores the shared attributes every consumer
    relies on (``source_lang_nllb``, ``target_lang_nllb``, ``df``,
    ``df_train``, ``df_validate``) and performs the train/validate split.

    Subclasses that must not split themselves override ``split()``.
    """

    def __init__(
        self, source_lang_nllb: str, target_lang_nllb: str, df: pd.DataFrame
    ) -> None:
        self.source_lang_nllb = source_lang_nllb
        self.target_lang_nllb = target_lang_nllb
        self.df = df
        self.df_train, self.df_validate = self.split(df)

    def split(self, df: pd.DataFrame):
        return train_test_split(df, test_size=0.02, random_state=SPLIT_SEED)


class TatoebaCorpus(BaseParallelCorpus):
    """Parallel corpus created from Tatoeba.

    Unlike ParallelFileCorpus, splitting is **deferred**. The constructor loads
    the data (including Tatoeba sentence IDs) but does NOT split.  Splitting
    happens later in ``main_corpus`` via ``_global_tatoeba_split`` so that
    a single, globally consistent set of held-out sentence IDs is used across
    all Tatoeba language pairs.
    """

    def __init__(self, sl_tat, tl_tat, sl_nllb, tl_nllb, cfg: RunConfig):
        main_download([sl_tat, tl_tat], redownload=False, tatoeba_path=cfg.tatoeba_path)
        df = load_tatoeba(sl_tat, tl_tat, cfg=cfg)
        super().__init__(sl_nllb, tl_nllb, df)

    def split(self, df: pd.DataFrame):
        """No local split: ``_global_tatoeba_split`` assigns the sets later."""
        return None, None


class ParallelFileCorpus(BaseParallelCorpus):
    """One configured CSV/TSV file represented as one corpus object.

    The language pair is read from the file's own headers, see
    ``load_parallel_table``.
    """

    def __init__(self, path: str, separator: str | None = None):
        self.path = path
        df, source_lang_nllb, target_lang_nllb = load_parallel_table(
            path, sep=separator
        )
        super().__init__(source_lang_nllb, target_lang_nllb, df)


def _normalize_for_overlap(text: str) -> str:
    return " ".join(text.split()).casefold()


def _remove_tatoeba_validation_overlap(
    corpus: ParallelFileCorpus, tatoeba_corpora: list[TatoebaCorpus]
) -> int:
    """Remove parallel training pairs that contain held-out Tatoeba text.

    Matching is language-aware and ignores whitespace and casing differences.
    The parallel corpus's own validation set is left unchanged.
    """
    held_out_by_language: dict[str, set[str]] = {}
    for tatoeba_corpus in tatoeba_corpora:
        if tatoeba_corpus.df_validate is None:
            raise RuntimeError(
                "Tatoeba corpora must be split before overlap filtering."
            )
        for language, column in (
            (tatoeba_corpus.source_lang_nllb, "source_sentence"),
            (tatoeba_corpus.target_lang_nllb, "target_sentence"),
        ):
            held_out_by_language.setdefault(language, set()).update(
                tatoeba_corpus.df_validate[column].map(_normalize_for_overlap)
            )

    source_overlap = (
        corpus.df_train["source_sentence"]
        .map(_normalize_for_overlap)
        .isin(held_out_by_language.get(corpus.source_lang_nllb, set()))
    )
    target_overlap = (
        corpus.df_train["target_sentence"]
        .map(_normalize_for_overlap)
        .isin(held_out_by_language.get(corpus.target_lang_nllb, set()))
    )
    overlap = source_overlap | target_overlap
    removed = int(overlap.sum())
    if removed:
        corpus.df_train = corpus.df_train.loc[~overlap].reset_index(drop=True)
    return removed


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

    src_df = pd.read_csv(
        src_file, sep="\t", header=None, names=["id", "lang", "source_sentence"]
    )
    trg_df = pd.read_csv(
        trg_file, sep="\t", header=None, names=["id", "lang", "target_sentence"]
    )
    link = pd.read_csv(
        link_file, sep="\t", header=None, names=["origin", "translation"]
    )

    df = link.merge(trg_df, left_on="origin", right_on="id").merge(
        src_df, left_on="translation", right_on="id"
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

    print(
        f"Global Tatoeba split: {len(all_ids_sorted):,} unique sentence IDs, "
        f"{holdout_n:,} ({GLOBAL_HOLDOUT_FRACTION:.0%}) held out for validation"
    )

    # Step 3 — split each corpus
    sentence_cols = ["source_sentence", "target_sentence"]
    for corpus in tatoeba_corpora:
        is_val = corpus.df["src_id"].isin(holdout_ids) | corpus.df["trg_id"].isin(
            holdout_ids
        )
        corpus.df_validate = corpus.df.loc[is_val, sentence_cols].reset_index(drop=True)
        corpus.df_train = corpus.df.loc[~is_val, sentence_cols].reset_index(drop=True)

        pair = f"{corpus.source_lang_nllb}-{corpus.target_lang_nllb}"
        total = len(corpus.df)
        n_val = len(corpus.df_validate)
        print(
            f"  {pair}: {total:,} total -> {n_val:,} val ({100 * n_val / total:.1f}%), "
            f"{total - n_val:,} train"
        )

        # Step 4 — drop IDs from df too (no longer needed)
        corpus.df = corpus.df[sentence_cols].reset_index(drop=True)


def main_corpus(
    source_langs_tatoeba,
    source_langs_nllb,
    parallel_data_paths=None,
    recursive: bool = True,
    parallel_data_separator: str | None = None,
    cfg: RunConfig | None = None,
):
    """
    Builds:
      • 1 corpus per Tatoeba language pair
      • 1 corpus per configured parallel CSV/TSV file

    Tatoeba corpora are split with a **global sentence-ID hold-out** so that
    no sentence appearing in any validation set can also appear in any
    training set (across all language pairs).  See module docstring for
    rationale.
    """

    cfg = cfg or get_default_config()
    if parallel_data_separator is None:
        parallel_data_separator = cfg.parallel_data_separator

    # Tatoeba corpora (split is deferred)
    tatoeba_corpora: list[TatoebaCorpus] = []
    zipped = list(zip(source_langs_tatoeba, source_langs_nllb))

    for i, (sl_tat, sl_nllb) in enumerate(zipped):
        for tl_tat, tl_nllb in zipped[i + 1 :]:
            print(f"Setting up Tatoeba corpus for {sl_nllb} - {tl_nllb}")
            tatoeba_corpora.append(
                TatoebaCorpus(sl_tat, tl_tat, sl_nllb, tl_nllb, cfg=cfg)
            )

    # Perform the global split across all Tatoeba corpora at once
    _global_tatoeba_split(tatoeba_corpora)

    corpora: list[BaseParallelCorpus] = list(tatoeba_corpora)

    # Configured parallel corpora (independent split, no Tatoeba IDs)
    if parallel_data_paths:
        files = find_parallel_files(parallel_data_paths, recursive=recursive)
        if not files:
            print(f"No parallel data files found in: {parallel_data_paths}")
        else:
            for f in files:
                print(f"Loading parallel data file: {f}")
                parallel_corpus = ParallelFileCorpus(
                    f, separator=parallel_data_separator
                )
                removed = _remove_tatoeba_validation_overlap(
                    parallel_corpus, tatoeba_corpora
                )
                if removed:
                    print(
                        f"  Removed {removed:,} training pairs overlapping "
                        "Tatoeba validation text."
                    )
                corpora.append(parallel_corpus)
    else:
        print("No additional parallel data paths provided.")

    return corpora


def pool_parallel_data_into_tatoeba(
    corpora: list[BaseParallelCorpus],
) -> list[BaseParallelCorpus]:
    """Pool additional parallel training data into matching Tatoeba corpora.

    For each ParallelFileCorpus, finds the first TatoebaCorpus with the same
    language pair and appends its ``df_train`` rows to it.
    Returns a new list containing only the (now-enlarged) Tatoeba corpora.

    This avoids temperature-sampling problems where tiny parallel files
    get their own sampling slots and are oversampled dozens of times.
    After pooling, all data of the same language pair shares one slot.

    **Use for training only.** For evaluation, pass the original unpooled
    list, so per-file metrics are still reported separately.
    """
    tatoeba: list[BaseParallelCorpus] = [
        c for c in corpora if isinstance(c, TatoebaCorpus)
    ]
    parallel_files = [c for c in corpora if isinstance(c, ParallelFileCorpus)]

    for pc in parallel_files:
        key = frozenset([pc.source_lang_nllb, pc.target_lang_nllb])
        match = next(
            (
                tc
                for tc in tatoeba
                if frozenset([tc.source_lang_nllb, tc.target_lang_nllb]) == key
            ),
            None,
        )
        if match is None:
            print(
                f"  Warning: no matching Tatoeba corpus for {pc.source_lang_nllb}-"
                f"{pc.target_lang_nllb}, keeping parallel file as standalone corpus."
            )
            tatoeba.append(pc)
        else:
            before = len(match.df_train)
            match.df_train = pd.concat(
                [match.df_train, pc.df_train],
                ignore_index=True,
            )
            print(
                f"  Pooled parallel file ({len(pc.df_train):,} rows) into "
                f"{match.source_lang_nllb}-{match.target_lang_nllb}: "
                f"{before:,} -> {len(match.df_train):,} train rows"
            )

    return tatoeba
