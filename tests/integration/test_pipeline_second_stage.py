from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel

from adminlineage.io import read_jsonl
from adminlineage.llm_types import BaseLLMClient
from adminlineage.pipeline import run_pipeline


class SecondStageClient(BaseLLMClient):
    def __init__(
        self,
        *,
        first_pass_links: dict[str, dict[str, Any]] | None = None,
        research_by_primary_key: dict[str, dict[str, Any]] | None = None,
        decision_by_primary_key: dict[str, dict[str, Any]] | None = None,
        error_on_research: set[str] | None = None,
        error_on_decision: set[str] | None = None,
    ) -> None:
        self.first_pass_links = first_pass_links or {}
        self.research_by_primary_key = research_by_primary_key or {}
        self.decision_by_primary_key = decision_by_primary_key or {}
        self.error_on_research = error_on_research or set()
        self.error_on_decision = error_on_decision or set()
        self.first_pass_calls = 0
        self.research_calls: list[str] = []
        self.decision_calls: list[str] = []
        self.research_search_flags: list[bool] = []
        self.decision_search_flags: list[bool] = []

    def generate_json(
        self,
        prompt: str,
        schema: Any,
        model: str,
        temperature: float,
        seed: int,
        *,
        enable_google_search: bool = False,
    ) -> dict[str, Any]:
        _ = (model, temperature, seed, enable_google_search)
        payload = json.loads(prompt.split("INPUT_PAYLOAD_JSON:\n", maxsplit=1)[1].strip())

        if "items" in payload:
            self.first_pass_calls += 1
            include_evidence = bool(payload.get("include_evidence", False))
            include_reason = bool(payload.get("include_reason", False))
            decisions: list[dict[str, Any]] = []
            for item in payload["items"]:
                configured = self.first_pass_links.get(item["from_key"])
                if configured is None:
                    link = {
                        "to_key": None,
                        "link_type": "unknown",
                        "relationship": "unknown",
                        "score": 0.2,
                    }
                else:
                    link = dict(configured)
                if include_evidence and "evidence" not in link:
                    link["evidence"] = "First-pass test response."
                if include_reason and "reason" not in link:
                    link["reason"] = "First-pass test response."
                decisions.append({"from_key": item["from_key"], "links": [link]})
            response = {"decisions": decisions}
            if isinstance(schema, type) and issubclass(schema, BaseModel):
                return schema.model_validate(response).model_dump()
            return response

        primary_key = payload["primary_item"]["primary_key"]
        if "candidate_subset" not in payload:
            self.research_calls.append(primary_key)
            self.research_search_flags.append(enable_google_search)
            if primary_key in self.error_on_research:
                raise RuntimeError(f"research failed for {primary_key}")
            response = self.research_by_primary_key.get(
                primary_key,
                {
                    "event_type": "unknown",
                    "lineage_hint": "",
                    "notes": "No strong lineage signal found.",
                },
            )
            if isinstance(schema, type) and issubclass(schema, BaseModel):
                return schema.model_validate(response).model_dump()
            return response

        self.decision_calls.append(primary_key)
        self.decision_search_flags.append(enable_google_search)
        if primary_key in self.error_on_decision:
            raise RuntimeError(f"decision failed for {primary_key}")
        response = self.decision_by_primary_key.get(
            primary_key,
            {
                "primary_key": primary_key,
                "selected_secondary_keys": [],
                "link_type": "no_match",
                "relationship": "unknown",
                "score": 0.0,
            },
        )
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            return schema.model_validate(response).model_dump()
        return response

    def generate_text(
        self,
        prompt: str,
        model: str,
        temperature: float,
        seed: int,
        *,
        enable_google_search: bool = False,
    ) -> str:
        _ = (prompt, model, temperature, seed, enable_google_search)
        raise NotImplementedError


def _run_dir(base_dir: Path, map_col: str = "district") -> Path:
    return base_dir / f"india_1951_2001_{map_col}"


def test_second_stage_is_disabled_when_exact_match_prune_is_none(tmp_path: Path):
    df_from = pd.DataFrame({"district": ["Faizabad"], "unit_id": ["f1"]})
    df_to = pd.DataFrame({"district": ["Ayodhya"], "unit_id": ["t1"]})
    client = SecondStageClient()

    crosswalk, metadata = run_pipeline(
        df_from,
        df_to,
        country="India",
        year_from=1951,
        year_to=2001,
        map_col_from="district",
        map_col_to="district",
        id_col_from="unit_id",
        id_col_to="unit_id",
        string_exact_match_prune="none",
        output_dir=tmp_path / "outputs_none",
        llm_client=client,
        output_write_parquet=False,
    )

    assert metadata["counts"]["second_stage_attempted_rows"] == 0
    assert metadata["counts"]["second_stage_rewritten_rows"] == 0
    assert metadata["counts"]["second_stage_failed_rows"] == 0
    assert "second_stage_results_jsonl" not in metadata["artifacts"]
    assert crosswalk["lineage_hint"].fillna("").eq("").all()


def test_second_stage_skips_decision_call_for_unknown_research_without_lineage_hint(
    tmp_path: Path,
):
    df_from = pd.DataFrame({"district": ["Faizabad"], "unit_id": ["f1"]})
    df_to = pd.DataFrame({"district": ["Ayodhya"], "unit_id": ["t1"]})
    client = SecondStageClient(
        first_pass_links={
            "from_0": {
                "to_key": None,
                "link_type": "unknown",
                "relationship": "unknown",
                "score": 0.2,
            }
        },
        research_by_primary_key={
            "from_0": {
                "event_type": "unknown",
                "lineage_hint": "",
                "notes": "Search did not surface a reliable lineage clue.",
            }
        },
    )

    crosswalk, metadata = run_pipeline(
        df_from,
        df_to,
        country="India",
        year_from=1951,
        year_to=2001,
        map_col_from="district",
        map_col_to="district",
        id_col_from="unit_id",
        id_col_to="unit_id",
        string_exact_match_prune="from",
        output_dir=tmp_path / "outputs_skip_decision",
        llm_client=client,
        output_write_parquet=False,
    )

    assert client.research_calls == ["from_0"]
    assert client.research_search_flags == [True]
    assert client.decision_calls == []
    assert client.decision_search_flags == []
    unmatched_row = crosswalk.loc[crosswalk["merge"] == "only_in_from"].iloc[0]
    assert unmatched_row["lineage_hint"] == ""
    assert metadata["counts"]["second_stage_attempted_rows"] == 1
    assert metadata["counts"]["second_stage_rewritten_rows"] == 0
    records = read_jsonl(
        _run_dir(tmp_path / "outputs_skip_decision") / "second_stage_results.jsonl"
    )
    assert len(records) == 1
    assert records[0]["status"] == "ok"
    assert records[0]["decision"]["selected_secondary_keys"] == []


def test_second_stage_from_rescues_only_in_from_rows_and_writes_lineage_hint(tmp_path: Path):
    df_from = pd.DataFrame({"district": ["Faizabad"], "unit_id": ["f1"]})
    df_to = pd.DataFrame({"district": ["Ayodhya"], "unit_id": ["t1"]})
    client = SecondStageClient(
        research_by_primary_key={
            "from_0": {
                "event_type": "rename",
                "lineage_hint": "Ayodhya",
                "notes": "Faizabad district was renamed to Ayodhya.",
            }
        },
        decision_by_primary_key={
            "from_0": {
                "primary_key": "from_0",
                "selected_secondary_keys": ["to_0"],
                "link_type": "rename",
                "relationship": "father_to_father",
                "score": 0.96,
            }
        },
    )

    crosswalk, metadata = run_pipeline(
        df_from,
        df_to,
        country="India",
        year_from=1951,
        year_to=2001,
        map_col_from="district",
        map_col_to="district",
        id_col_from="unit_id",
        id_col_to="unit_id",
        string_exact_match_prune="from",
        output_dir=tmp_path / "outputs_from_rescue",
        llm_client=client,
        output_write_parquet=False,
    )

    matched = crosswalk.loc[crosswalk["merge"] == "both"].reset_index(drop=True)
    assert len(matched) == 1
    assert matched.loc[0, "from_name"] == "Faizabad"
    assert matched.loc[0, "to_name"] == "Ayodhya"
    assert matched.loc[0, "lineage_hint"] == "Ayodhya"
    assert client.research_search_flags == [True]
    assert client.decision_search_flags == [False]
    assert metadata["counts"]["second_stage_attempted_rows"] == 1
    assert metadata["counts"]["second_stage_rewritten_rows"] == 1
    assert metadata["counts"]["second_stage_failed_rows"] == 0


def test_second_stage_to_rescues_only_in_to_rows(tmp_path: Path):
    df_from = pd.DataFrame({"district": ["Allahabad"], "unit_id": ["f1"]})
    df_to = pd.DataFrame({"district": ["Prayagraj"], "unit_id": ["t1"]})
    client = SecondStageClient(
        research_by_primary_key={
            "to_0": {
                "event_type": "rename",
                "lineage_hint": "Allahabad",
                "notes": "Allahabad was renamed to Prayagraj.",
            }
        },
        decision_by_primary_key={
            "to_0": {
                "primary_key": "to_0",
                "selected_secondary_keys": ["from_0"],
                "link_type": "rename",
                "relationship": "father_to_father",
                "score": 0.95,
            }
        },
    )

    crosswalk, metadata = run_pipeline(
        df_from,
        df_to,
        country="India",
        year_from=1951,
        year_to=2001,
        map_col_from="district",
        map_col_to="district",
        id_col_from="unit_id",
        id_col_to="unit_id",
        string_exact_match_prune="to",
        output_dir=tmp_path / "outputs_to_rescue",
        llm_client=client,
        output_write_parquet=False,
    )

    matched = crosswalk.loc[crosswalk["merge"] == "both"].reset_index(drop=True)
    assert len(matched) == 1
    assert matched.loc[0, "from_name"] == "Allahabad"
    assert matched.loc[0, "to_name"] == "Prayagraj"
    assert matched.loc[0, "lineage_hint"] == "Allahabad"
    assert metadata["counts"]["second_stage_rewritten_rows"] == 1


def test_second_stage_supports_row_per_link_many_to_one_rewrites(tmp_path: Path):
    df_from = pd.DataFrame(
        {
            "district": ["North Tehsil", "South Tehsil"],
            "unit_id": ["f1", "f2"],
        }
    )
    df_to = pd.DataFrame({"district": ["Unified Tehsil"], "unit_id": ["t1"]})
    client = SecondStageClient(
        research_by_primary_key={
            "to_0": {
                "event_type": "merge",
                "lineage_hint": "North Tehsil",
                "notes": "Unified Tehsil absorbed North and South Tehsil.",
            }
        },
        decision_by_primary_key={
            "to_0": {
                "primary_key": "to_0",
                "selected_secondary_keys": ["from_0", "from_1"],
                "link_type": "merge",
                "relationship": "father_to_father",
                "score": 0.91,
            }
        },
    )

    crosswalk, metadata = run_pipeline(
        df_from,
        df_to,
        country="India",
        year_from=1951,
        year_to=2001,
        map_col_from="district",
        map_col_to="district",
        id_col_from="unit_id",
        id_col_to="unit_id",
        string_exact_match_prune="to",
        output_dir=tmp_path / "outputs_many_to_one",
        llm_client=client,
        output_write_parquet=False,
    )

    matched = crosswalk.loc[crosswalk["merge"] == "both"].sort_values("from_name")
    assert matched["to_name"].tolist() == ["Unified Tehsil", "Unified Tehsil"]
    assert matched["link_type"].tolist() == ["merge", "merge"]
    assert metadata["counts"]["second_stage_rewritten_rows"] == 1


def test_second_stage_uses_global_secondary_pool_even_when_exact_match_scope_differs(
    tmp_path: Path,
):
    df_from = pd.DataFrame(
        {
            "state": ["Old State"],
            "district": ["Allahabad"],
            "unit_id": ["f1"],
        }
    )
    df_to = pd.DataFrame(
        {
            "state": ["New State"],
            "district": ["Prayagraj"],
            "unit_id": ["t1"],
        }
    )
    client = SecondStageClient(
        research_by_primary_key={
            "to_0": {
                "event_type": "rename",
                "lineage_hint": "Allahabad",
                "notes": "Renamed district record found outside the first-pass scope.",
            }
        },
        decision_by_primary_key={
            "to_0": {
                "primary_key": "to_0",
                "selected_secondary_keys": ["from_0"],
                "link_type": "rename",
                "relationship": "father_to_father",
                "score": 0.9,
            }
        },
    )

    crosswalk, _ = run_pipeline(
        df_from,
        df_to,
        country="India",
        year_from=1951,
        year_to=2001,
        map_col_from="district",
        map_col_to="district",
        exact_match=["state"],
        id_col_from="unit_id",
        id_col_to="unit_id",
        string_exact_match_prune="to",
        output_dir=tmp_path / "outputs_global_secondary",
        llm_client=client,
        output_write_parquet=False,
    )

    matched_row = crosswalk.loc[crosswalk["merge"] == "both"].iloc[0]
    assert matched_row["from_name"] == "Allahabad"
    assert matched_row["to_name"] == "Prayagraj"
    assert matched_row["constraints_passed"]["exact_match"] is False


def test_second_stage_resume_retries_only_rows_with_error_records(tmp_path: Path):
    df_from = pd.DataFrame({"district": ["Faizabad"], "unit_id": ["f1"]})
    df_to = pd.DataFrame({"district": ["Ayodhya"], "unit_id": ["t1"]})
    output_dir = tmp_path / "outputs_resume"

    failing_client = SecondStageClient(
        research_by_primary_key={
            "from_0": {
                "event_type": "rename",
                "lineage_hint": "Ayodhya",
                "notes": "Research succeeded before decision failed.",
            }
        },
        error_on_decision={"from_0"},
    )
    first_crosswalk, first_metadata = run_pipeline(
        df_from,
        df_to,
        country="India",
        year_from=1951,
        year_to=2001,
        map_col_from="district",
        map_col_to="district",
        id_col_from="unit_id",
        id_col_to="unit_id",
        string_exact_match_prune="from",
        output_dir=output_dir,
        llm_client=failing_client,
        output_write_parquet=False,
    )
    first_lineage_hint = first_crosswalk.loc[
        first_crosswalk["merge"] == "only_in_from",
        "lineage_hint",
    ].iloc[0]
    assert first_lineage_hint == ""
    assert first_metadata["counts"]["second_stage_failed_rows"] == 1

    succeeding_client = SecondStageClient(
        research_by_primary_key={
            "from_0": {
                "event_type": "rename",
                "lineage_hint": "Ayodhya",
                "notes": "Faizabad was renamed to Ayodhya.",
            }
        },
        decision_by_primary_key={
            "from_0": {
                "primary_key": "from_0",
                "selected_secondary_keys": ["to_0"],
                "link_type": "rename",
                "relationship": "father_to_father",
                "score": 0.94,
            }
        },
    )
    second_crosswalk, second_metadata = run_pipeline(
        df_from,
        df_to,
        country="India",
        year_from=1951,
        year_to=2001,
        map_col_from="district",
        map_col_to="district",
        id_col_from="unit_id",
        id_col_to="unit_id",
        string_exact_match_prune="from",
        output_dir=output_dir,
        llm_client=succeeding_client,
        output_write_parquet=False,
    )

    assert succeeding_client.first_pass_calls == 0
    assert succeeding_client.research_calls == ["from_0"]
    assert succeeding_client.decision_calls == ["from_0"]
    rescued_lineage_hint = second_crosswalk.loc[
        second_crosswalk["merge"] == "both",
        "lineage_hint",
    ].iloc[0]
    assert rescued_lineage_hint == "Ayodhya"
    assert second_metadata["counts"]["second_stage_failed_rows"] == 0
    assert second_metadata["counts"]["second_stage_rewritten_rows"] == 1


def test_second_stage_results_are_restored_from_replay_bundle(tmp_path: Path):
    df_from = pd.DataFrame({"district": ["Faizabad"], "unit_id": ["f1"]})
    df_to = pd.DataFrame({"district": ["Ayodhya"], "unit_id": ["t1"]})
    replay_store_dir = tmp_path / "shared-replay"
    first_output_dir = tmp_path / "outputs_first"
    second_output_dir = tmp_path / "outputs_second"

    first_client = SecondStageClient(
        research_by_primary_key={
            "from_0": {
                "event_type": "rename",
                "lineage_hint": "Ayodhya",
                "notes": "Faizabad was renamed to Ayodhya.",
            }
        },
        decision_by_primary_key={
            "from_0": {
                "primary_key": "from_0",
                "selected_secondary_keys": ["to_0"],
                "link_type": "rename",
                "relationship": "father_to_father",
                "score": 0.95,
            }
        },
    )
    first_crosswalk, first_metadata = run_pipeline(
        df_from,
        df_to,
        country="India",
        year_from=1951,
        year_to=2001,
        map_col_from="district",
        map_col_to="district",
        id_col_from="unit_id",
        id_col_to="unit_id",
        string_exact_match_prune="from",
        output_dir=first_output_dir,
        llm_client=first_client,
        replay_enabled=True,
        replay_store_dir=replay_store_dir,
        output_write_parquet=False,
    )

    second_client = SecondStageClient()
    second_crosswalk, second_metadata = run_pipeline(
        df_from,
        df_to,
        country="India",
        year_from=1951,
        year_to=2001,
        map_col_from="district",
        map_col_to="district",
        id_col_from="unit_id",
        id_col_to="unit_id",
        string_exact_match_prune="from",
        output_dir=second_output_dir,
        llm_client=second_client,
        replay_enabled=True,
        replay_store_dir=replay_store_dir,
        output_write_parquet=False,
    )

    assert second_client.first_pass_calls == 0
    assert second_client.research_calls == []
    assert second_client.decision_calls == []
    assert second_metadata["execution_mode"] == "replay"
    assert second_metadata["replay_hit"] is True
    pd.testing.assert_frame_equal(first_crosswalk, second_crosswalk)

    run_dir = _run_dir(second_output_dir)
    second_stage_records = read_jsonl(run_dir / "second_stage_results.jsonl")
    assert second_stage_records
    assert second_stage_records[0]["primary_key"] == "from_0"


def test_second_stage_skipped_decision_results_are_restored_from_replay_bundle(tmp_path: Path):
    df_from = pd.DataFrame({"district": ["Faizabad"], "unit_id": ["f1"]})
    df_to = pd.DataFrame({"district": ["Ayodhya"], "unit_id": ["t1"]})
    replay_store_dir = tmp_path / "shared-replay-skip"
    first_output_dir = tmp_path / "outputs_first_skip"
    second_output_dir = tmp_path / "outputs_second_skip"

    first_client = SecondStageClient(
        first_pass_links={
            "from_0": {
                "to_key": None,
                "link_type": "unknown",
                "relationship": "unknown",
                "score": 0.2,
            }
        },
        research_by_primary_key={
            "from_0": {
                "event_type": "unknown",
                "lineage_hint": "",
                "notes": "No reliable lineage clue surfaced.",
            }
        },
    )
    first_crosswalk, first_metadata = run_pipeline(
        df_from,
        df_to,
        country="India",
        year_from=1951,
        year_to=2001,
        map_col_from="district",
        map_col_to="district",
        id_col_from="unit_id",
        id_col_to="unit_id",
        string_exact_match_prune="from",
        output_dir=first_output_dir,
        llm_client=first_client,
        replay_enabled=True,
        replay_store_dir=replay_store_dir,
        output_write_parquet=False,
    )

    second_client = SecondStageClient()
    second_crosswalk, second_metadata = run_pipeline(
        df_from,
        df_to,
        country="India",
        year_from=1951,
        year_to=2001,
        map_col_from="district",
        map_col_to="district",
        id_col_from="unit_id",
        id_col_to="unit_id",
        string_exact_match_prune="from",
        output_dir=second_output_dir,
        llm_client=second_client,
        replay_enabled=True,
        replay_store_dir=replay_store_dir,
        output_write_parquet=False,
    )

    assert first_metadata["counts"]["second_stage_attempted_rows"] == 1
    assert second_client.first_pass_calls == 0
    assert second_client.research_calls == []
    assert second_client.decision_calls == []
    assert second_metadata["execution_mode"] == "replay"
    assert second_metadata["replay_hit"] is True
    pd.testing.assert_frame_equal(first_crosswalk, second_crosswalk)

    run_dir = _run_dir(second_output_dir)
    second_stage_records = read_jsonl(run_dir / "second_stage_results.jsonl")
    assert len(second_stage_records) == 1
    assert second_stage_records[0]["decision"]["selected_secondary_keys"] == []
