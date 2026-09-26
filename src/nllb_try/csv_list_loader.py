import os
import re
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
_SINGLE_CAPS_WORD_FRAGMENT = re.compile(
    r"(?P<word>[^\W\d_]+)(?P<suffix>[\d\s\W_]*)", re.UNICODE
)
_REPEATED_LAYOUT_OR_DIGIT = re.compile(r"([0-9_—=\-*,])\1{3,}")
_DOCUMENT_METADATA_WORDS = frozenset(
    {
        "inhoud",
        "inhold",
        "pagina",
        "bladzijde",
        "bladzie",
        "bladziede",
        "isbn",
        "redactie",
        "redaksie",
        "colofon",
    }
)


def _is_single_caps_word_fragment(text: str) -> bool:
    match = _SINGLE_CAPS_WORD_FRAGMENT.fullmatch(text.strip())
    return bool(match and match.group("word").isupper())


def _contains_low_quality_pair(source: str, target: str) -> bool:
    """Identify clearly non-linguistic or document-layout parallel pairs."""
    if not any(character.isalpha() for character in source) or not any(
        character.isalpha() for character in target
    ):
        return True

    is_caps_fragment = _is_single_caps_word_fragment(
        source
    ) or _is_single_caps_word_fragment(target)
    words = {
        word.casefold()
        for text in (source, target)
        for word in _WORD.findall(text)
    }
    if is_caps_fragment and (
        words & _DOCUMENT_METADATA_WORDS
        or any(character.isdigit() for text in (source, target) for character in text)
    ):
        return True

    has_repeated_layout_or_digit = bool(
        _REPEATED_LAYOUT_OR_DIGIT.search(source)
        or _REPEATED_LAYOUT_OR_DIGIT.search(target)
    )
    return (
        has_repeated_layout_or_digit
        and len(_WORD.findall(source)) <= 1
        and len(_WORD.findall(target)) <= 1
    )


def _clean_df(df: pd.DataFrame, source_col: str, target_col: str) -> pd.DataFrame:
    """Keep complete, useful pairs and normalize their column names.

    Whitespace is only used to identify blank values. Clearly non-linguistic
    rows and document-layout fragments are removed, but retained sentence text
    is not stripped or otherwise modified.
    """
    df = df.dropna(subset=[source_col, target_col])
    df = df[df[source_col].str.strip().str.len() > 0]
    df = df[df[target_col].str.strip().str.len() > 0]
    keep = [
        not _contains_low_quality_pair(source, target)
        for source, target in zip(df[source_col], df[target_col], strict=True)
    ]
    df = df.loc[keep]
    df = df[[source_col, target_col]].copy()
    df = df.rename(
        columns={source_col: "source_sentence", target_col: "target_sentence"}
    )
    return df


def load_parallel_table(
    path: str | os.PathLike[str], sep: str | None = None
) -> tuple[pd.DataFrame, str, str]:
    """Load a two-column parallel file using NLLB language-code headers.

    The first header is the source language and the second header is the
    target language. Headers must look like e.g. ``gos_Latn``. Files must be
    UTF-8; a UTF-8 byte-order mark is accepted. Empty and whitespace-only
    pairs and clearly non-linguistic/layout pairs are discarded. Other
    sentence text is preserved verbatim.

    Unless ``sep`` is provided, ``.csv`` files use ``;`` and ``.tsv`` files
    use a tab.
    """
    path = Path(path)
    if sep is None:
        separators = {".csv": ";", ".tsv": "\t"}
        try:
            sep = separators[path.suffix.lower()]
        except KeyError as error:
            raise ValueError(
                f"Cannot infer a separator for {path}; expected .csv or .tsv, "
                "or pass sep explicitly."
            ) from error

    nllb_language_label = re.compile(r"^[a-z]{3}_[A-Z][a-z]{3}$")
    try:
        df_raw = pd.read_csv(
            path,
            sep=sep,
            header=0,
            encoding="utf-8-sig",
            dtype="string",
        )
    except UnicodeDecodeError as error:
        raise ValueError(f"{path}: parallel data must be UTF-8 encoded.") from error
    except pd.errors.ParserError as error:
        raise ValueError(
            f"{path}: could not parse parallel data using separator {sep!r}."
        ) from error

    columns = [str(column).strip() for column in df_raw.columns]

    if len(columns) != 2:
        raise ValueError(
            f"{path}: expected exactly two columns with NLLB language labels "
            f"(for example nld_Latn{sep}gos_Latn) using separator {sep!r}, "
            f"found {columns!r}."
        )
    if len(set(columns)) != 2 or not all(
        nllb_language_label.fullmatch(column) for column in columns
    ):
        raise ValueError(
            f"{path}: expected exactly two columns with NLLB language labels "
            f"(for example nld_Latn{sep}gos_Latn), found {columns!r}."
        )

    df_raw.columns = columns
    source_lang, target_lang = columns
    return _clean_df(df_raw, source_lang, target_lang), source_lang, target_lang


def find_parallel_files(
    paths: str | os.PathLike[str] | Sequence[str | os.PathLike[str]],
    recursive: bool = True,
) -> list[str]:
    """Expand configured files and directories into sorted CSV/TSV paths.

    Directories are searched recursively by default. Explicit files must have
    a ``.csv`` or ``.tsv`` extension; other files inside directories are
    ignored.
    """
    parallel_file_extensions = (".csv", ".tsv")
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]

    found: set[str] = set()
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"Parallel data path does not exist: {path}")
        if path.is_file():
            if path.suffix.lower() not in parallel_file_extensions:
                raise ValueError(
                    f"Unsupported parallel data file {path}; expected .csv or .tsv."
                )
            found.add(str(path))
            continue
        if not path.is_dir():
            raise ValueError(
                f"Parallel data path is neither a file nor directory: {path}"
            )

        iterator = path.rglob("*") if recursive else path.glob("*")
        found.update(
            str(candidate)
            for candidate in iterator
            if candidate.is_file()
            and candidate.suffix.lower() in parallel_file_extensions
        )

    return sorted(found)
