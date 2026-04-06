from __future__ import annotations

import pandas as pd

from adminlineage.validation import collapse_duplicate_match_keys, validate_inputs_data


def test_validate_inputs_success(sample_df_from, sample_df_to):
    result = validate_inputs_data(
        sample_df_from,
        sample_df_to,
        country="India",
        map_col_from="subdistrict",
        map_col_to="subdistrict",
        exact_match=["state", "district"],
        id_col_from="unit_id",
        id_col_to="unit_id",
    )
    assert result["valid"] is True
    assert result["errors"] == []


def test_validate_inputs_missing_col(sample_df_from, sample_df_to):
    result = validate_inputs_data(
        sample_df_from,
        sample_df_to,
        country="India",
        map_col_from="missing",
        map_col_to="subdistrict",
        exact_match=["state", "district"],
        id_col_from=None,
        id_col_to=None,
    )
    assert result["valid"] is False
    assert any("map_col_from" in err for err in result["errors"])


def test_validate_inputs_warns_without_anchors(sample_df_from, sample_df_to):
    result = validate_inputs_data(
        sample_df_from,
        sample_df_to,
        country="India",
        map_col_from="subdistrict",
        map_col_to=None,
        exact_match=None,
        id_col_from=None,
        id_col_to=None,
    )
    assert result["valid"] is True
    assert result["map_col_to"] == "subdistrict"
    assert result["warnings"]


def test_validate_inputs_reports_duplicate_collapse_counts():
    df_from = pd.DataFrame(
        {
            "state": ["S1", "S1", "S1"],
            "district": ["D1", "D1", "D2"],
            "subdistrict": ["North Block", "North Block", "River Tehsil"],
            "unit_id": ["f1", "f2", "f3"],
        }
    )
    df_to = pd.DataFrame(
        {
            "state": ["S1", "S1", "S1"],
            "district": ["D1", "D1", "D2"],
            "subdistrict": ["North Block", "North Block", "River Tehsil"],
            "unit_id": ["t1", "t2", "t3"],
        }
    )

    result = validate_inputs_data(
        df_from,
        df_to,
        country="India",
        map_col_from="subdistrict",
        map_col_to="subdistrict",
        exact_match=["state", "district"],
        id_col_from="unit_id",
        id_col_to="unit_id",
    )

    assert result["valid"] is True
    assert result["from_rows_input"] == 3
    assert result["from_rows_effective"] == 2
    assert result["to_rows_input"] == 3
    assert result["to_rows_effective"] == 2
    assert any("Collapsed 1 duplicate df_from rows" in warning for warning in result["warnings"])
    assert any("Collapsed 1 duplicate df_to rows" in warning for warning in result["warnings"])


def test_validate_inputs_collapses_name_only_duplicates_without_exact_match():
    df_from = pd.DataFrame(
        {
            "subdistrict": ["Alpha", "Alpha", "Beta"],
            "unit_id": ["f1", "f2", "f3"],
        }
    )
    df_to = pd.DataFrame(
        {
            "subdistrict": ["Gamma", "Gamma", "Delta"],
            "unit_id": ["t1", "t2", "t3"],
        }
    )

    result = validate_inputs_data(
        df_from,
        df_to,
        country="India",
        map_col_from="subdistrict",
        map_col_to="subdistrict",
        exact_match=None,
        id_col_from="unit_id",
        id_col_to="unit_id",
    )

    assert result["valid"] is True
    assert result["from_rows_effective"] == 2
    assert result["to_rows_effective"] == 2
    assert any("No exact_match columns supplied" in warning for warning in result["warnings"])
    assert any("Collapsed 1 duplicate df_from rows" in warning for warning in result["warnings"])
    assert any("Collapsed 1 duplicate df_to rows" in warning for warning in result["warnings"])


def test_collapse_duplicate_match_keys_keeps_first_row():
    df = pd.DataFrame(
        {
            "state": ["S1", "S1", "S1"],
            "district": ["D1", "D1", "D2"],
            "subdistrict": ["North Block", "North Block", "River Tehsil"],
            "unit_id": ["keep-me", "drop-me", "keep-too"],
        }
    )

    collapsed, report = collapse_duplicate_match_keys(
        df,
        key_cols=["state", "district", "subdistrict"],
        side_label="df_from",
    )

    assert report["collapsed"] is True
    assert report["collapsed_rows"] == 1
    assert collapsed["unit_id"].tolist() == ["keep-me", "keep-too"]


def test_collapse_duplicate_match_keys_normalizes_string_columns():
    df = pd.DataFrame(
        {
            "state": [" State A ", "state a", "State B"],
            "district": ["DELHI", "delhi", "Mumbai"],
            "unit_id": ["keep-me", "drop-me", "keep-too"],
        }
    )

    collapsed, report = collapse_duplicate_match_keys(
        df,
        key_cols=["state", "district"],
        side_label="df_from",
    )

    assert report["collapsed"] is True
    assert report["collapsed_rows"] == 1
    assert collapsed["unit_id"].tolist() == ["keep-me", "keep-too"]
