"""Cheap candidate generation utilities."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True, slots=True)
class TargetCandidateRecord:
    """Compact target-row view for repeated shortlist generation."""

    to_key: str
    to_name: Any
    to_canonical_name: str
    to_tokens: set[str]
    to_char_ngrams: Mapping[str, int]


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


def prepare_target_records(to_group: pd.DataFrame) -> list[TargetCandidateRecord]:
    """Materialize reusable target records for repeated shortlist scoring."""

    if to_group.empty:
        return []

    column_index = {name: idx for idx, name in enumerate(to_group.columns)}
    return [
        TargetCandidateRecord(
            to_key=row[column_index["_to_key"]],
            to_name=row[column_index["_to_name_raw"]],
            to_canonical_name=row[column_index["_to_canonical_name"]],
            to_tokens=row[column_index["_to_tokens"]],
            to_char_ngrams=row[column_index["_to_char_ngrams"]],
        )
        for row in to_group.itertuples(index=False, name=None)
    ]


def generate_shortlist_from_records(
    from_tokens: set[str],
    from_ngrams: Mapping[str, int],
    target_records: list[TargetCandidateRecord],
    *,
    max_candidates: int,
) -> list[dict[str, Any]]:
    """Generate ranked candidate list from precomputed target records."""

    if not target_records:
        return []

    ranked: list[dict[str, Any]] = []
    for target in target_records:
        token_score = token_jaccard(from_tokens, target.to_tokens)
        ngram_score = ngram_cosine(from_ngrams, target.to_char_ngrams)
        base_score = combined_similarity(token_score, ngram_score)

        ranked.append(
            {
                "to_key": target.to_key,
                "to_name": target.to_name,
                "to_canonical_name": target.to_canonical_name,
                "score": float(round(base_score, 6)),
                "token_jaccard": float(round(token_score, 6)),
                "ngram_cosine": float(round(ngram_score, 6)),
            }
        )

    ranked.sort(key=lambda item: (-item["score"], item["to_canonical_name"], item["to_key"]))
    return ranked[:max_candidates]


def generate_shortlist(
    from_row: pd.Series,
    to_group: pd.DataFrame,
    *,
    max_candidates: int,
) -> list[dict[str, Any]]:
    """Generate ranked candidate list for one from-row."""

    return generate_shortlist_from_records(
        from_row["_from_tokens"],
        from_row["_from_char_ngrams"],
        prepare_target_records(to_group),
        max_candidates=max_candidates,
    )
