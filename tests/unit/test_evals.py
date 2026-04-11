from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import adminlineage
from adminlineage import api
from adminlineage.evals import evaluate_crosswalk as evaluate_crosswalk_impl


def test_api_evaluate_crosswalk_forwards_arguments(monkeypatch):
    captured: dict[str, object] = {}

    def fake_evaluate_crosswalk(crosswalk, ground_truth, **kwargs):
        captured["crosswalk"] = crosswalk
        captured["ground_truth"] = ground_truth
        captured["kwargs"] = kwargs
        return {"summary": {"f1": 1.0}}

    monkeypatch.setattr(api, "_evaluate_crosswalk", fake_evaluate_crosswalk)

    crosswalk = pd.DataFrame({"from_canonical_name": ["agra"], "to_canonical_name": ["agra"]})
    truth = pd.DataFrame({"district_old": ["Agra"], "district_new": ["Agra"]})

    result = api.evaluate_crosswalk(
        crosswalk,
        truth,
        truth_from_col="district_old",
        truth_to_col="district_new",
        truth_scope_map={"state_old": "state"},
        predicted_merge_values=("both", "only_in_from"),
        normalize_strings=False,
        drop_duplicates=False,
    )

    assert result == {"summary": {"f1": 1.0}}
    assert captured["crosswalk"] is crosswalk
    assert captured["ground_truth"] is truth
    assert captured["kwargs"] == {
        "truth_from_col": "district_old",
        "truth_to_col": "district_new",
        "truth_scope_map": {"state_old": "state"},
        "predicted_from_col": "from_canonical_name",
        "predicted_to_col": "to_canonical_name",
        "predicted_merge_values": ("both", "only_in_from"),
        "normalize_strings": False,
        "drop_duplicates": False,
    }


def test_package_exports_evaluate_crosswalk():
    assert callable(adminlineage.evaluate_crosswalk)


def test_evaluate_crosswalk_deduplicates_truth_rows_and_scores_matches():
    crosswalk = pd.DataFrame(
        {
            "from_canonical_name": ["agra", "kanpur dehat", "bareilly"],
            "to_canonical_name": ["agra", "kanpur rural", "bareilly"],
            "merge": ["both", "both", "only_in_from"],
            "state": ["uttar pradesh", "uttar pradesh", "uttar pradesh"],
        }
    )
    truth = pd.DataFrame(
        {
            "state_old": ["Uttar Pradesh", "Uttar Pradesh", "Uttar Pradesh"],
            "district_old": ["Agra", "Kanpur Dehat", "Kanpur Dehat"],
            "state_new": ["Uttar Pradesh", "Uttar Pradesh", "Uttar Pradesh"],
            "district_new": ["Agra", "Kanpur Rural", "Kanpur Rural"],
        }
    )

    result = evaluate_crosswalk_impl(
        crosswalk,
        truth,
        truth_from_col="district_old",
        truth_to_col="district_new",
        truth_scope_map={"state_old": "state"},
    )

    assert result["summary"] == {
        "predicted_links": 2,
        "ground_truth_links": 2,
        "true_positive_links": 2,
        "false_positive_links": 0,
        "false_negative_links": 0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
    }
    assert list(result["true_positives"].index.names) == [
        "eval_from_norm",
        "eval_to_norm",
        "eval_scope_state_old_norm",
    ]
    assert result["false_positives"].empty
    assert result["false_negatives"].empty


def test_evaluate_crosswalk_uses_paths_and_merge_filter(tmp_path: Path):
    crosswalk = pd.DataFrame(
        {
            "from_canonical_name": ["agra", "faizabad", "allahabad"],
            "to_canonical_name": ["agra", "ayodhya", "prayagraj"],
            "merge": ["both", "both", "only_in_from"],
            "state": ["uttar pradesh", "uttar pradesh", "uttar pradesh"],
        }
    )
    truth = pd.DataFrame(
        {
            "state_old": ["Uttar Pradesh", "Uttar Pradesh", "Uttar Pradesh"],
            "district_old": ["Agra", "Faizabad", "Allahabad"],
            "state_new": ["Uttar Pradesh", "Uttar Pradesh", "Uttar Pradesh"],
            "district_new": ["Agra", "Ayodhya", "Prayagraj"],
        }
    )
    crosswalk_path = tmp_path / "evolution_key.csv"
    truth_path = tmp_path / "ground_truth.csv"
    crosswalk.to_csv(crosswalk_path, index=False)
    truth.to_csv(truth_path, index=False)

    result = evaluate_crosswalk_impl(
        crosswalk_path,
        truth_path,
        truth_from_col="district_old",
        truth_to_col="district_new",
        truth_scope_map={"state_old": "state"},
    )

    assert result["summary"]["predicted_links"] == 2
    assert result["summary"]["ground_truth_links"] == 3
    assert result["summary"]["true_positive_links"] == 2
    assert result["summary"]["false_negative_links"] == 1
    assert "allahabad" in set(result["false_negatives"]["eval_from_norm"])


def test_evaluate_crosswalk_scope_mapping_prevents_cross_state_false_match():
    crosswalk = pd.DataFrame(
        {
            "from_canonical_name": ["aurangabad"],
            "to_canonical_name": ["aurangabad"],
            "merge": ["both"],
            "state": ["bihar"],
        }
    )
    truth = pd.DataFrame(
        {
            "state_old": ["Maharashtra"],
            "district_old": ["Aurangabad"],
            "district_new": ["Aurangabad"],
        }
    )

    result = evaluate_crosswalk_impl(
        crosswalk,
        truth,
        truth_from_col="district_old",
        truth_to_col="district_new",
        truth_scope_map={"state_old": "state"},
    )

    assert result["summary"]["true_positive_links"] == 0
    assert result["summary"]["false_positive_links"] == 1
    assert result["summary"]["false_negative_links"] == 1


def test_evaluate_crosswalk_respects_normalize_strings_toggle():
    crosswalk = pd.DataFrame(
        {
            "from_canonical_name": ["Prayagraj."],
            "to_canonical_name": ["Ayodhya"],
            "merge": ["both"],
        }
    )
    truth = pd.DataFrame(
        {
            "district_old": ["prayagraj"],
            "district_new": ["ayodhya"],
        }
    )

    normalized = evaluate_crosswalk_impl(
        crosswalk,
        truth,
        truth_from_col="district_old",
        truth_to_col="district_new",
    )
    raw = evaluate_crosswalk_impl(
        crosswalk,
        truth,
        truth_from_col="district_old",
        truth_to_col="district_new",
        normalize_strings=False,
    )

    assert normalized["summary"]["true_positive_links"] == 1
    assert raw["summary"]["true_positive_links"] == 0


def test_evaluate_crosswalk_raises_for_missing_required_columns():
    crosswalk = pd.DataFrame({"from_canonical_name": ["agra"], "merge": ["both"]})
    truth = pd.DataFrame({"district_old": ["Agra"], "district_new": ["Agra"]})

    with pytest.raises(
        ValueError,
        match="crosswalk is missing required columns: to_canonical_name",
    ):
        evaluate_crosswalk_impl(
            crosswalk,
            truth,
            truth_from_col="district_old",
            truth_to_col="district_new",
        )
