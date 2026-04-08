from __future__ import annotations

import pandas as pd

from adminlineage.replay import build_replay_identity
from adminlineage.schema import get_output_schema_definition, normalize_nullable_output_columns


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
        "model": "gemini-3.1-flash-lite-preview",
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


def test_normalize_nullable_output_columns_converts_string_dtype_targets_to_object():
    crosswalk = pd.DataFrame(
        {
            "from_name": pd.Series(["North Block", pd.NA], dtype="string"),
            "from_canonical_name": pd.Series(["north block", pd.NA], dtype="string"),
            "from_id": pd.Series(["f1", pd.NA], dtype="string"),
            "from_key": pd.Series(["from_0", pd.NA], dtype="string"),
            "to_key": pd.Series(["to_0", pd.NA], dtype="string"),
            "to_name": pd.Series(["North Block", pd.NA], dtype="string"),
            "to_canonical_name": pd.Series(["north block", pd.NA], dtype="string"),
            "to_id": pd.Series(["t1", pd.NA], dtype="string"),
        }
    )

    normalized = normalize_nullable_output_columns(crosswalk)

    assert normalized["from_name"].dtype == object
    assert normalized["from_canonical_name"].dtype == object
    assert normalized["from_id"].dtype == object
    assert normalized["from_key"].dtype == object
    assert normalized["to_key"].dtype == object
    assert normalized["to_name"].dtype == object
    assert normalized["to_canonical_name"].dtype == object
    assert normalized["to_id"].dtype == object
    assert normalized.loc[1, "from_name"] is None
    assert normalized.loc[1, "from_canonical_name"] is None
    assert normalized.loc[1, "from_id"] is None
    assert normalized.loc[1, "from_key"] is None
    assert normalized.loc[1, "to_key"] is None
    assert normalized.loc[1, "to_name"] is None
    assert normalized.loc[1, "to_canonical_name"] is None
    assert normalized.loc[1, "to_id"] is None


def test_output_schema_definition_includes_merge_column_and_enum():
    schema = get_output_schema_definition(include_evidence=False)

    assert "merge" in schema["crosswalk_columns"]
    assert "merge" in schema["required_output_columns"]
    assert "lineage_hint" in schema["crosswalk_columns"]
    assert "lineage_hint" in schema["required_output_columns"]
    assert schema["merge_enum"] == ["both", "only_in_from", "only_in_to"]
