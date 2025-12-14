import os
import pandas as pd

def _clean_df(df: pd.DataFrame, nl_col: str, var_col: str) -> pd.DataFrame:
    df = df.dropna(subset=[nl_col, var_col])
    df = df[df[nl_col].str.strip().str.len() > 2]
    df = df[df[var_col].str.strip().str.len() > 1]
    df = df[[nl_col, var_col]].copy()
    df = df.rename(columns={nl_col: "nld_Latn", var_col: "gos_Latn"})
    df["src_lang"] = "nld_Latn"
    df["tgt_lang"] = "gos_Latn"
    return df

def load_parallel_table(path: str, sep: str = ";") -> pd.DataFrame:
    df_raw = pd.read_csv(path, sep=sep, header=0, encoding='utf-8-sig')

    invalid = [c for c in df_raw.columns if not isinstance(c, str) or c.strip() == "" or c.strip().startswith("Unnamed:")]
    valid   = [c for c in df_raw.columns if c not in invalid]

    for c in invalid:
        df_raw = df_raw[df_raw[c].isna() | (df_raw[c].astype(str).str.strip() == "")]

    if not valid:
        raise ValueError(f"{path}: no usable columns found.")

    nl_cols = [c for c in valid if "nederland" in c.lower()]
    if len(nl_cols) != 1:
        raise ValueError(f"{path}: expected exactly one NL column.")
    nl_col = nl_cols[0]

    variety_cols = [c for c in valid if c != nl_col]
    if len(variety_cols) != 1:
        raise ValueError(f"{path}: expected exactly one variety column.")
    var_col = variety_cols[0]

    return _clean_df(df_raw, nl_col, var_col)

def find_variety_files(base_dir: str, recursive: bool = True) -> list[str]:
    """Zoek alle .csv en .tsv bestanden in base_dir (+ subfolders als recursive=True)."""

    exts = (".csv", ".tsv")
    found = []

    if recursive:
        for root, _, files in os.walk(base_dir):
            for f in files:
                if f.lower().endswith(exts):
                    found.append(os.path.join(root, f))
    else:
        for f in os.listdir(base_dir):
            if f.lower().endswith(exts):
                found.append(os.path.join(base_dir, f))

    return found