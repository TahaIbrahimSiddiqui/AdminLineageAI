"""Name normalization and feature extraction for administrative units."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

import pandas as pd


_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_SPACE_RE = re.compile(r"\s+")


def canonicalize_name(name: str) -> str:
    """Normalize place names for cheap lexical matching."""

    if name is None:
        return ""
    value = unicodedata.normalize("NFKC", str(name))
    value = value.lower().strip()
    value = _PUNCT_RE.sub(" ", value)
    value = _SPACE_RE.sub(" ", value)
    return value.strip()


def tokenize(name: str) -> set[str]:
    """Tokenize canonical names into unique tokens."""

    if not name:
        return set()
    return {token for token in name.split(" ") if token}


def char_ngram_counter(name: str, n: int = 3) -> Counter[str]:
    """Count character n-grams for cosine-like similarity."""

    if not name:
        return Counter()
    padded = f"  {name} "
    grams = [padded[i : i + n] for i in range(max(len(padded) - n + 1, 0))]
    return Counter(grams)


def add_normalized_columns(df: pd.DataFrame, name_col: str, prefix: str) -> pd.DataFrame:
    """Return copy of DataFrame with canonicalization helper columns."""

    if name_col not in df.columns:
        raise ValueError(f"Missing required column: {name_col}")

    out = df.copy()
    out[f"_{prefix}_name_raw"] = out[name_col].astype(str)
    out[f"_{prefix}_canonical_name"] = out[f"_{prefix}_name_raw"].map(canonicalize_name)
    out[f"_{prefix}_tokens"] = out[f"_{prefix}_canonical_name"].map(tokenize)
    out[f"_{prefix}_char_ngrams"] = out[f"_{prefix}_canonical_name"].map(char_ngram_counter)
    return out
