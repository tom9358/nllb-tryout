import os
import re
from collections.abc import Sequence
from pathlib import Path

import pandas as pd


def _clean_df(df: pd.DataFrame, source_col: str, target_col: str) -> pd.DataFrame:
    """Keep complete, non-blank pairs and normalize their column names.

    Whitespace is only used to identify blank values; sentence text itself is
    not stripped or otherwise modified.
    """
    df = df.dropna(subset=[source_col, target_col])
    df = df[df[source_col].str.strip().str.len() > 0]
    df = df[df[target_col].str.strip().str.len() > 0]
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
    pairs are discarded, but non-empty sentence text is preserved verbatim.

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
