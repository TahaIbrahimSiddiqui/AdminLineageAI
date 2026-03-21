from __future__ import annotations

from adminlineage.validation import validate_inputs_data


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
