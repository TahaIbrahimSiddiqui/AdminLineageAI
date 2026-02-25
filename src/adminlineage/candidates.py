"""Cheap candidate generation utilities."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping
from typing import Any

import pandas as pd

from .normalize import canonicalize_name


AliasLookup = dict[tuple[str, tuple[Any, ...] | None], set[str]]


def token_jaccard(left: set[str], right: set[str]) -> float:
    """Compute Jaccard similarity over token sets."""

    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    intersection = len(left & right)
    union = len(left | right)
    return intersection / union if union else 0.0


def ngram_cosine(left: Mapping[str, int], right: Mapping[str, int]) -> float:
    """Compute cosine similarity over character n-gram counters."""

    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0

    dot = 0.0
    for key, value in left.items():
        dot += value * right.get(key, 0)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def combined_similarity(
    token_score: float,
    ngram_score: float,
    token_weight: float = 0.55,
    ngram_weight: float = 0.45,
) -> float:
    """Weighted lexical similarity score in [0,1]."""

    value = token_weight * token_score + ngram_weight * ngram_score
    return max(0.0, min(1.0, value))


def build_alias_lookup(aliases: pd.DataFrame | None, anchor_cols: list[str]) -> AliasLookup:
    """Build alias lookup keyed by canonical from-name and optional anchor tuple."""

    lookup: AliasLookup = defaultdict(set)
    if aliases is None or aliases.empty:
        return {}

    required = {"from_alias", "to_alias"}
    missing = sorted(required - set(aliases.columns))
    if missing:
        raise ValueError(f"aliases DataFrame missing required columns: {missing}")

    alias_df = aliases.copy()
    alias_df["_from_alias_c"] = alias_df["from_alias"].astype(str).map(canonicalize_name)
    alias_df["_to_alias_c"] = alias_df["to_alias"].astype(str).map(canonicalize_name)

    for _, row in alias_df.iterrows():
        anchor_values: tuple[Any, ...] | None = None
        if anchor_cols and all(col in alias_df.columns for col in anchor_cols):
            anchor_values = tuple(row[col] for col in anchor_cols)
        key = (row["_from_alias_c"], anchor_values)
        lookup[key].add(row["_to_alias_c"])

        # Add global fallback mapping for this alias.
        global_key = (row["_from_alias_c"], None)
        lookup[global_key].add(row["_to_alias_c"])

    return dict(lookup)


def generate_shortlist(
    from_row: pd.Series,
    to_group: pd.DataFrame,
    *,
    max_candidates: int,
    alias_lookup: AliasLookup,
    anchor_cols: list[str],
) -> list[dict[str, Any]]:
    """Generate ranked candidate list for one from-row."""

    if to_group.empty:
        return []

    from_canonical = from_row["_from_canonical_name"]
    from_tokens = from_row["_from_tokens"]
    from_ngrams = from_row["_from_char_ngrams"]

    anchor_tuple: tuple[Any, ...] | None = None
    if anchor_cols:
        anchor_tuple = tuple(from_row[col] for col in anchor_cols)

    alias_targets = set()
    alias_targets.update(alias_lookup.get((from_canonical, None), set()))
    alias_targets.update(alias_lookup.get((from_canonical, anchor_tuple), set()))

    ranked: list[dict[str, Any]] = []
    for _, to_row in to_group.iterrows():
        token_score = token_jaccard(from_tokens, to_row["_to_tokens"])
        ngram_score = ngram_cosine(from_ngrams, to_row["_to_char_ngrams"])
        base_score = combined_similarity(token_score, ngram_score)
        alias_hit = to_row["_to_canonical_name"] in alias_targets
        score = max(base_score, 0.95) if alias_hit else base_score

        ranked.append(
            {
                "to_key": to_row["_to_key"],
                "to_name": to_row["_to_name_raw"],
                "to_canonical_name": to_row["_to_canonical_name"],
                "score": float(round(score, 6)),
                "token_jaccard": float(round(token_score, 6)),
                "ngram_cosine": float(round(ngram_score, 6)),
                "alias_hit": bool(alias_hit),
            }
        )

    ranked.sort(key=lambda item: (-item["score"], item["to_canonical_name"], item["to_key"]))
    return ranked[:max_candidates]
