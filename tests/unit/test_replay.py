from __future__ import annotations

import pandas as pd

from adminlineage.replay import build_replay_identity


def _build_identity(
    df_from: pd.DataFrame,
    df_to: pd.DataFrame,
    **request_overrides,
) -> dict[str, object]:
    request_payload = {
        "country": "India",
        "year_from": 1951,
        "year_to": 2001,
        "exact_match": ["state", "district"],
        "map_col_from": "subdistrict",
        "map_col_to": "subdistrict",
        "relationship": "auto",
        "string_exact_match_prune": "from",
        "reason": False,
        "model": "gemini-2.5-pro",
        "batch_size": 10,
        "max_candidates": 5,
        "seed": 42,
        "temperature": 0.75,
        "enable_google_search": True,
        "schema_version": "2.1.0",
    }
    request_payload.update(request_overrides)
    return build_replay_identity(
        request_payload=request_payload,
        llm_backend="GeminiClient",
        df_from=df_from,
        df_to=df_to,
        map_col_from="subdistrict",
        map_col_to="subdistrict",
        exact_match=["state", "district"],
        id_col_from="unit_id",
        id_col_to="unit_id",
        extra_context_cols=["context_note"],
    )


def test_replay_key_is_stable_when_row_order_changes():
    df_from = pd.DataFrame(
        {
            "state": ["S1", "S1"],
            "district": ["D1", "D2"],
            "subdistrict": ["North Block", "River Tehsil"],
            "unit_id": ["f1", "f2"],
            "context_note": ["alpha", "beta"],
        }
    )
    df_to = pd.DataFrame(
        {
            "state": ["S1", "S1"],
            "district": ["D1", "D2"],
            "subdistrict": ["North Block", "River Tehsil"],
            "unit_id": ["t1", "t2"],
            "context_note": ["alpha", "beta"],
        }
    )

    original = _build_identity(df_from, df_to)
    reordered = _build_identity(
        df_from.iloc[::-1].reset_index(drop=True),
        df_to.iloc[::-1].reset_index(drop=True),
    )

    assert original["replay_key"] == reordered["replay_key"]
    assert original["from_fingerprint"] == reordered["from_fingerprint"]
    assert original["to_fingerprint"] == reordered["to_fingerprint"]


def test_replay_key_changes_when_semantic_request_changes():
    df_from = pd.DataFrame(
        {
            "state": ["S1"],
            "district": ["D1"],
            "subdistrict": ["North Block"],
            "unit_id": ["f1"],
            "context_note": ["alpha"],
        }
    )
    df_to = pd.DataFrame(
        {
            "state": ["S1"],
            "district": ["D1"],
            "subdistrict": ["North Block"],
            "unit_id": ["t1"],
            "context_note": ["alpha"],
        }
    )

    baseline = _build_identity(df_from, df_to)
    changed = _build_identity(df_from, df_to, relationship="father_to_child")

    assert baseline["replay_key"] != changed["replay_key"]
