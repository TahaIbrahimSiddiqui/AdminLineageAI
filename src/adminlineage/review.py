"""Global consistency checks and review-queue generation."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd


def coverage_summary(crosswalk: pd.DataFrame, exact_match: list[str]) -> dict[str, dict[str, int]]:
    """Compute coverage counts by exact-match group."""

    if crosswalk.empty:
        return {}

    grouped: dict[str, dict[str, int]] = {}
    group_iter: Iterable[tuple[Any, pd.DataFrame]]
    if exact_match:
        group_iter = crosswalk.groupby(exact_match, dropna=False)
    else:
        group_iter = [("__all__", crosswalk)]

    for key, group in group_iter:
        label = str(key)
        total_from = group["from_key"].nunique()
        matched_from = group[
            (group["to_key"].notna())
            & (group["link_type"].isin(["rename", "split", "merge", "transfer"]))
        ]["from_key"].nunique()
        grouped[label] = {
            "from_units": int(total_from),
            "matched_from_units": int(matched_from),
            "unmatched_from_units": int(total_from - matched_from),
        }

    return grouped


def apply_global_flags(
    crosswalk: pd.DataFrame,
    *,
    low_score_threshold: float,
    max_fan_out: int = 3,
    max_fan_in: int = 3,
) -> pd.DataFrame:
    """Add suspicious-pattern flags per row for downstream review."""

    if crosswalk.empty:
        return crosswalk.copy()

    out = crosswalk.copy()
    matched_mask = out["to_key"].notna() & out["from_key"].notna()
    if "merge" in out.columns:
        matched_mask &= out["merge"].eq("both")
    else:
        matched_mask &= out["link_type"].isin(["rename", "split", "merge", "transfer"])
    matched_rows = out[matched_mask]
    fan_out = matched_rows.groupby("from_key")["to_key"].nunique().to_dict()
    fan_in = matched_rows.groupby("to_key")["from_key"].nunique().to_dict()

    flags_column: list[list[str]] = []
    for _, row in out.iterrows():
        flags: list[str] = []
        if row["score"] < low_score_threshold:
            flags.append("low_score")
        if row["link_type"] in {"unknown", "no_match"}:
            flags.append("type_needs_review")
        if fan_out.get(row["from_key"], 0) > max_fan_out:
            flags.append("high_fan_out")
        if pd.notna(row["to_key"]) and fan_in.get(row["to_key"], 0) > max_fan_in:
            flags.append("high_fan_in")
        flags_column.append(flags)

    out["review_flags"] = flags_column
    out["review_reason"] = out["review_flags"].map(lambda vals: ",".join(vals))
    return out


def build_review_queue(crosswalk_with_flags: pd.DataFrame) -> pd.DataFrame:
    """Extract rows that should be manually reviewed."""

    if crosswalk_with_flags.empty:
        return crosswalk_with_flags.copy()

    mask = crosswalk_with_flags["review_flags"].map(bool)
    review = crosswalk_with_flags[mask].copy()
    return review.sort_values(["from_name", "score"], ascending=[True, True]).reset_index(drop=True)


def summarize_counts(crosswalk: pd.DataFrame) -> dict[str, Any]:
    """Compute simple run counts for run metadata."""

    if crosswalk.empty:
        return {
            "rows": 0,
            "from_units": 0,
            "to_units": 0,
            "avg_score": None,
        }

    return {
        "rows": int(len(crosswalk)),
        "from_units": int(crosswalk["from_key"].nunique()),
        "to_units": int(crosswalk["to_key"].dropna().nunique()),
        "avg_score": float(round(crosswalk["score"].mean(), 6)),
    }
