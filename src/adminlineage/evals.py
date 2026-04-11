"""Utilities for comparing crosswalk outputs against human ground truth."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from .io import read_dataframe
from .normalize import canonicalize_name
from .utils import sanitize_name


def evaluate_crosswalk(
    crosswalk: pd.DataFrame | str | Path,
    ground_truth: pd.DataFrame | str | Path,
    *,
    truth_from_col: str,
    truth_to_col: str,
    truth_scope_map: dict[str, str] | None = None,
    predicted_from_col: str = "from_canonical_name",
    predicted_to_col: str = "to_canonical_name",
    predicted_merge_values: tuple[str, ...] = ("both",),
    normalize_strings: bool = True,
    drop_duplicates: bool = True,
) -> dict[str, Any]:
    """Compare a materialized crosswalk against a human-maintained truth table."""

    scope_pairs = list((truth_scope_map or {}).items())
    predicted = _prepare_predicted_links(
        crosswalk,
        predicted_from_col=predicted_from_col,
        predicted_to_col=predicted_to_col,
        scope_pairs=scope_pairs,
        predicted_merge_values=predicted_merge_values,
        normalize_strings=normalize_strings,
        drop_duplicates=drop_duplicates,
    )
    truth = _prepare_truth_links(
        ground_truth,
        truth_from_col=truth_from_col,
        truth_to_col=truth_to_col,
        scope_pairs=scope_pairs,
        normalize_strings=normalize_strings,
        drop_duplicates=drop_duplicates,
    )

    key_col = "eval_key"
    normalized_cols = [
        "eval_from_norm",
        "eval_to_norm",
        *[f"eval_scope_{_scope_suffix(truth_col)}_norm" for truth_col, _ in scope_pairs],
    ]

    predicted_counts = Counter(predicted[key_col].tolist())
    truth_counts = Counter(truth[key_col].tolist())
    tp_counts = Counter(
        {
            key: min(predicted_counts[key], truth_counts[key])
            for key in predicted_counts.keys() & truth_counts.keys()
        }
    )
    fp_counts = predicted_counts - truth_counts
    fn_counts = truth_counts - predicted_counts

    true_positive_predicted = _select_rows_by_counts(predicted, key_col=key_col, counts=tp_counts)
    true_positive_truth = _select_rows_by_counts(truth, key_col=key_col, counts=tp_counts)
    false_positive_rows = _select_rows_by_counts(predicted, key_col=key_col, counts=fp_counts)
    false_negative_rows = _select_rows_by_counts(truth, key_col=key_col, counts=fn_counts)

    true_positive_predicted = _with_match_order(true_positive_predicted, key_col=key_col)
    true_positive_truth = _with_match_order(true_positive_truth, key_col=key_col)
    true_positives = true_positive_predicted.merge(
        true_positive_truth,
        on=[key_col, "eval_match_index", *normalized_cols],
        how="inner",
    ).drop(columns=["eval_match_index"])

    tp_count = int(sum(tp_counts.values()))
    fp_count = int(sum(fp_counts.values()))
    fn_count = int(sum(fn_counts.values()))
    predicted_count = int(sum(predicted_counts.values()))
    truth_count = int(sum(truth_counts.values()))
    precision = tp_count / predicted_count if predicted_count else 0.0
    recall = tp_count / truth_count if truth_count else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision > 0.0 and recall > 0.0
        else 0.0
    )

    return {
        "summary": {
            "predicted_links": predicted_count,
            "ground_truth_links": truth_count,
            "true_positive_links": tp_count,
            "false_positive_links": fp_count,
            "false_negative_links": fn_count,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
        "true_positives": _set_eval_index(true_positives, normalized_cols),
        "false_positives": _set_eval_index(false_positive_rows, normalized_cols),
        "false_negatives": _set_eval_index(false_negative_rows, normalized_cols),
    }


def _load_frame(data: pd.DataFrame | str | Path, *, label: str) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    try:
        return read_dataframe(data)
    except Exception as exc:  # pragma: no cover - exercised through read_dataframe
        raise ValueError(f"Could not load {label}: {exc}") from exc


def _scope_suffix(column_name: str) -> str:
    return sanitize_name(column_name).replace("-", "_")


def _normalize_eval_value(value: Any, *, normalize_strings: bool) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    if normalize_strings and isinstance(value, str):
        return canonicalize_name(value)
    return value


def _make_eval_key(frame: pd.DataFrame, normalized_columns: list[str]) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=object)
    return frame.loc[:, normalized_columns].apply(lambda row: tuple(row.tolist()), axis=1)


def _drop_incomplete_links(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    valid_mask = pd.Series(True, index=frame.index)
    for column in columns:
        series = frame[column]
        if series.dtype == object:
            valid_mask &= series.map(lambda value: value not in (None, ""))
        else:
            valid_mask &= series.notna()
    return frame.loc[valid_mask].copy()


def _dedupe_links(
    frame: pd.DataFrame,
    *,
    normalized_columns: list[str],
    drop_duplicates: bool,
) -> pd.DataFrame:
    if not drop_duplicates or frame.empty:
        return frame.reset_index(drop=True)
    return frame.drop_duplicates(subset=normalized_columns, keep="first").reset_index(drop=True)


def _prepare_predicted_links(
    crosswalk: pd.DataFrame | str | Path,
    *,
    predicted_from_col: str,
    predicted_to_col: str,
    scope_pairs: list[tuple[str, str]],
    predicted_merge_values: tuple[str, ...],
    normalize_strings: bool,
    drop_duplicates: bool,
) -> pd.DataFrame:
    frame = _load_frame(crosswalk, label="crosswalk")
    required_columns = [
        predicted_from_col,
        predicted_to_col,
        *[predicted_col for _, predicted_col in scope_pairs],
    ]
    _require_columns(frame, required_columns, label="crosswalk")

    if predicted_merge_values:
        if "merge" not in frame.columns:
            raise ValueError(
                "Crosswalk is missing the 'merge' column required for "
                "predicted_merge_values filtering."
            )
        frame = frame.loc[frame["merge"].isin(predicted_merge_values)].copy()

    selected: dict[str, Any] = {
        "predicted_from": frame[predicted_from_col],
        "predicted_to": frame[predicted_to_col],
    }
    normalized_columns = ["eval_from_norm", "eval_to_norm"]
    selected["eval_from_norm"] = frame[predicted_from_col].map(
        lambda value: _normalize_eval_value(value, normalize_strings=normalize_strings)
    )
    selected["eval_to_norm"] = frame[predicted_to_col].map(
        lambda value: _normalize_eval_value(value, normalize_strings=normalize_strings)
    )

    for truth_col, predicted_col in scope_pairs:
        suffix = _scope_suffix(truth_col)
        selected[f"predicted_scope_{suffix}"] = frame[predicted_col]
        norm_col = f"eval_scope_{suffix}_norm"
        selected[norm_col] = frame[predicted_col].map(
            lambda value: _normalize_eval_value(value, normalize_strings=normalize_strings)
        )
        normalized_columns.append(norm_col)

    prepared = pd.DataFrame(selected)
    prepared = _drop_incomplete_links(prepared, normalized_columns)
    prepared = _dedupe_links(
        prepared,
        normalized_columns=normalized_columns,
        drop_duplicates=drop_duplicates,
    )
    prepared["eval_key"] = _make_eval_key(prepared, normalized_columns)
    return prepared


def _prepare_truth_links(
    ground_truth: pd.DataFrame | str | Path,
    *,
    truth_from_col: str,
    truth_to_col: str,
    scope_pairs: list[tuple[str, str]],
    normalize_strings: bool,
    drop_duplicates: bool,
) -> pd.DataFrame:
    frame = _load_frame(ground_truth, label="ground_truth")
    required_columns = [truth_from_col, truth_to_col, *[truth_col for truth_col, _ in scope_pairs]]
    _require_columns(frame, required_columns, label="ground_truth")

    selected: dict[str, Any] = {
        "truth_from": frame[truth_from_col],
        "truth_to": frame[truth_to_col],
    }
    normalized_columns = ["eval_from_norm", "eval_to_norm"]
    selected["eval_from_norm"] = frame[truth_from_col].map(
        lambda value: _normalize_eval_value(value, normalize_strings=normalize_strings)
    )
    selected["eval_to_norm"] = frame[truth_to_col].map(
        lambda value: _normalize_eval_value(value, normalize_strings=normalize_strings)
    )

    for truth_col, _ in scope_pairs:
        suffix = _scope_suffix(truth_col)
        selected[f"truth_scope_{suffix}"] = frame[truth_col]
        norm_col = f"eval_scope_{suffix}_norm"
        selected[norm_col] = frame[truth_col].map(
            lambda value: _normalize_eval_value(value, normalize_strings=normalize_strings)
        )
        normalized_columns.append(norm_col)

    prepared = pd.DataFrame(selected)
    prepared = _drop_incomplete_links(prepared, normalized_columns)
    prepared = _dedupe_links(
        prepared,
        normalized_columns=normalized_columns,
        drop_duplicates=drop_duplicates,
    )
    prepared["eval_key"] = _make_eval_key(prepared, normalized_columns)
    return prepared


def _require_columns(frame: pd.DataFrame, columns: list[str], *, label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"{label} is missing required columns: {joined}")


def _select_rows_by_counts(
    frame: pd.DataFrame,
    *,
    key_col: str,
    counts: Counter[tuple[Any, ...]],
) -> pd.DataFrame:
    if frame.empty or not counts:
        return frame.iloc[0:0].copy()

    remaining = Counter(counts)
    keep_rows: list[bool] = []
    for key in frame[key_col]:
        if remaining[key] > 0:
            keep_rows.append(True)
            remaining[key] -= 1
        else:
            keep_rows.append(False)
    return frame.loc[keep_rows].copy().reset_index(drop=True)


def _with_match_order(frame: pd.DataFrame, *, key_col: str) -> pd.DataFrame:
    if frame.empty:
        out = frame.copy()
        out["eval_match_index"] = pd.Series(dtype=int)
        return out
    out = frame.copy()
    out["eval_match_index"] = out.groupby(key_col).cumcount()
    return out


def _set_eval_index(frame: pd.DataFrame, normalized_columns: list[str]) -> pd.DataFrame:
    if frame.empty:
        return frame.set_index(normalized_columns)
    return frame.set_index(normalized_columns, drop=False).sort_index()
