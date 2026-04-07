from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel

from adminlineage.llm.base import BaseLLMClient
from adminlineage.llm.gemini import GeminiClient
from adminlineage.pipeline import run_pipeline


class CapturingClient(BaseLLMClient):
    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.payloads: list[dict[str, Any]] = []
        self.search_flags: list[bool] = []
        self.text_prompts: list[str] = []
        self.text_payloads: list[dict[str, Any]] = []
        self.text_search_flags: list[bool] = []

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
        _ = (model, temperature, seed)
        self.prompts.append(prompt)
        self.search_flags.append(enable_google_search)

        marker = "INPUT_PAYLOAD_JSON:\n"
        payload = json.loads(prompt.split(marker, maxsplit=1)[1].strip())
        self.payloads.append(payload)

        decisions = []
        for item in payload["items"]:
            candidates = item.get("candidates", [])
            if candidates:
                link = {
                    "to_key": candidates[0]["to_key"],
                    "link_type": "rename",
                    "relationship": "father_to_father",
                    "score": 0.9,
                    "evidence": "Captured for prompt verification.",
                }
            else:
                link = {
                    "to_key": None,
                    "link_type": "no_match",
                    "relationship": "unknown",
                    "score": 0.0,
                    "evidence": "No candidates available.",
                }
            decisions.append({"from_key": item["from_key"], "links": [link]})

        response = {"decisions": decisions}
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
        _ = (model, temperature, seed)
        self.text_prompts.append(prompt)
        self.text_search_flags.append(enable_google_search)

        marker = "INPUT_PAYLOAD_JSON:\n"
        payload = json.loads(prompt.split(marker, maxsplit=1)[1].strip())
        self.text_payloads.append(payload)
        return "inconclusive"


class CapturingGeminiClient(GeminiClient):
    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.payloads: list[dict[str, Any]] = []
        self.search_flags: list[bool] = []

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
        _ = (model, temperature, seed)
        self.prompts.append(prompt)
        self.search_flags.append(enable_google_search)

        marker = "INPUT_PAYLOAD_JSON:\n"
        payload = json.loads(prompt.split(marker, maxsplit=1)[1].strip())
        self.payloads.append(payload)

        decisions = []
        for item in payload["items"]:
            candidates = item.get("candidates", [])
            if candidates:
                link = {
                    "to_key": candidates[0]["to_key"],
                    "link_type": "rename",
                    "relationship": "father_to_father",
                    "score": 0.9,
                    "evidence": "Captured for Gemini prompt verification.",
                }
            else:
                link = {
                    "to_key": None,
                    "link_type": "no_match",
                    "relationship": "unknown",
                    "score": 0.0,
                    "evidence": "No candidates available.",
                }
            decisions.append({"from_key": item["from_key"], "links": [link]})

        response = {"decisions": decisions}
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
        return "inconclusive"


class GroundedSequentialClient(BaseLLMClient):
    def __init__(self) -> None:
        self.json_payloads: list[dict[str, Any]] = []
        self.json_search_flags: list[bool] = []
        self.text_payloads: list[dict[str, Any]] = []
        self.text_search_flags: list[bool] = []

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
        _ = (model, temperature, seed)
        self.json_search_flags.append(enable_google_search)

        marker = "INPUT_PAYLOAD_JSON:\n"
        payload = json.loads(prompt.split(marker, maxsplit=1)[1].strip())
        self.json_payloads.append(payload)

        decisions = []
        for item in payload["items"]:
            if item["from_key"] == "from_0":
                link = {
                    "to_key": item["candidates"][0]["to_key"],
                    "link_type": "rename",
                    "relationship": "father_to_father",
                    "score": 0.95,
                    "evidence": "Confident shortlist match.",
                }
            elif enable_google_search:
                link = {
                    "to_key": item["candidates"][0]["to_key"],
                    "link_type": "rename",
                    "relationship": "father_to_father",
                    "score": 0.88,
                    "evidence": "Grounded verification supported the shortlist candidate.",
                }
            else:
                link = {
                    "to_key": None,
                    "link_type": "unknown",
                    "relationship": "unknown",
                    "score": 0.35,
                    "evidence": "Ambiguous without grounded verification.",
                }
            decisions.append({"from_key": item["from_key"], "links": [link]})

        response = {"decisions": decisions}
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
        _ = (model, temperature, seed)
        self.text_search_flags.append(enable_google_search)

        marker = "INPUT_PAYLOAD_JSON:\n"
        payload = json.loads(prompt.split(marker, maxsplit=1)[1].strip())
        self.text_payloads.append(payload)
        candidate_keys = [candidate["to_key"] for candidate in payload["candidate_subset"]]
        return (
            f"{candidate_keys[0]} is supported by search evidence for the administrative unit. "
            "No stronger contrary evidence was found."
        )


def test_pipeline_keeps_batch_adjudication_structured_when_search_is_enabled(
    sample_df_from,
    sample_df_to,
    tmp_path: Path,
):
    client = CapturingClient()

    run_pipeline(
        sample_df_from,
        sample_df_to,
        country="India",
        year_from=1951,
        year_to=2001,
        map_col_from="subdistrict",
        map_col_to="subdistrict",
        exact_match=["state", "district"],
        id_col_from="unit_id",
        id_col_to="unit_id",
        relationship="auto",
        reason=False,
        model="gemini-3.1-flash-lite-preview",
        batch_size=2,
        max_candidates=3,
        output_dir=tmp_path / "outputs_prompt_behavior",
        llm_client=client,
        temperature=1.0,
        enable_google_search=True,
        output_write_parquet=False,
    )

    assert client.search_flags == [True, True, True]
    assert client.text_search_flags == []
    assert client.prompts
    assert all(
        "If grounding tools are available, you may use them only to verify names, geography,"
        in prompt
        for prompt in client.prompts
    )
    assert all(
        "Search grounding is verification only." in prompt
        for prompt in client.prompts
    )


def test_pipeline_runs_single_pass_grounding_for_each_ai_row(
    tmp_path: Path,
):
    df_from = pd.DataFrame(
        {
            "district_name": ["Alpha District", "Beta District"],
            "unit_id": ["f1", "f2"],
        }
    )
    df_to = pd.DataFrame(
        {
            "District": ["Alpha District New", "Beta District New"],
            "unit_id": ["t1", "t2"],
        }
    )
    client = GroundedSequentialClient()

    crosswalk, metadata = run_pipeline(
        df_from,
        df_to,
        country="India",
        year_from=2011,
        year_to=2025,
        map_col_from="district_name",
        map_col_to="District",
        exact_match=[],
        id_col_from="unit_id",
        id_col_to="unit_id",
        relationship="auto",
        reason=False,
        model="gemini-3.1-flash-lite-preview",
        batch_size=2,
        max_candidates=3,
        output_dir=tmp_path / "outputs_prompt_behavior_grounded",
        llm_client=client,
        temperature=0.75,
        enable_google_search=True,
        output_write_parquet=False,
    )

    assert client.json_search_flags == [True, True]
    assert client.text_search_flags == []
    assert all(len(payload["items"]) == 1 for payload in client.json_payloads)
    assert metadata["counts"]["grounding_attempted_rows"] == 2
    assert metadata["counts"]["grounded_rows"] == 2
    assert metadata["counts"]["grounding_failed_rows"] == 0
    assert "grounding_notes_jsonl" in metadata["artifacts"]

    beta_row = crosswalk.loc[crosswalk["from_key"] == "from_1"].iloc[0]
    assert beta_row["link_type"] == "rename"
    assert beta_row["to_key"] == "to_1"


def test_pipeline_uses_sequential_grounded_json_calls_for_gemini_runs(tmp_path: Path):
    df_from = pd.DataFrame(
        {
            "district_name": [f"District {idx}" for idx in range(9)],
            "unit_id": [f"f{idx}" for idx in range(9)],
        }
    )
    df_to = pd.DataFrame(
        {
            "District": [f"District {idx}" for idx in range(12)],
            "unit_id": [f"t{idx}" for idx in range(12)],
        }
    )
    client = CapturingGeminiClient()

    _, metadata = run_pipeline(
        df_from,
        df_to,
        country="India",
        year_from=2011,
        year_to=2025,
        map_col_from="district_name",
        map_col_to="District",
        exact_match=[],
        id_col_from="unit_id",
        id_col_to="unit_id",
        relationship="auto",
        reason=False,
        model="gemini-3.1-flash-lite-preview",
        batch_size=25,
        max_candidates=15,
        output_dir=tmp_path / "outputs_large_batch_cap",
        llm_client=client,
        output_write_parquet=False,
    )

    assert client.payloads
    assert all(len(payload["items"]) == 1 for payload in client.payloads)
    assert any(
        "Using effective batch_size=1 instead." in warning for warning in metadata["warnings"]
    )


def test_string_exact_match_prune_from_skips_exact_from_rows_but_keeps_to_candidates(
    tmp_path: Path,
):
    df_from = pd.DataFrame(
        {
            "state": ["STATE A", "STATE A", "STATE B"],
            "district": ["DELHI", "Delhi Rural", "DELHI"],
            "unit_id": ["f1", "f2", "f3"],
        }
    )
    df_to = pd.DataFrame(
        {
            "state": ["state a", "state a", "state b"],
            "district": ["delhi", "delhi extension", "delhi"],
            "unit_id": ["t1", "t2", "t3"],
        }
    )
    client = CapturingClient()

    crosswalk, metadata = run_pipeline(
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
        relationship="auto",
        string_exact_match_prune="from",
        reason=False,
        model="gemini-3.1-flash-lite-preview",
        batch_size=10,
        max_candidates=5,
        output_dir=tmp_path / "outputs_prune_from",
        llm_client=client,
        output_write_parquet=False,
    )

    assert len(client.payloads) == 1
    payload = client.payloads[0]
    assert [item["from_name"] for item in payload["items"]] == ["Delhi Rural"]
    assert [candidate["to_name"] for candidate in payload["items"][0]["candidates"]] == [
        "delhi",
        "delhi extension",
    ]
    assert metadata["request"]["string_exact_match_prune"] == "from"

    exact_rows = crosswalk.loc[crosswalk["from_name"].str.upper() == "DELHI"]
    assert set(exact_rows["to_name"].str.lower()) == {"delhi"}
    assert exact_rows["score"].eq(1.0).all()


def test_string_exact_match_prune_to_removes_exact_to_candidates_from_ai_pool(tmp_path: Path):
    df_from = pd.DataFrame(
        {
            "state": ["STATE A", "STATE A", "STATE B"],
            "district": ["DELHI", "Delhi Rural", "DELHI"],
            "unit_id": ["f1", "f2", "f3"],
        }
    )
    df_to = pd.DataFrame(
        {
            "state": ["state a", "state a", "state b"],
            "district": ["delhi", "delhi extension", "delhi"],
            "unit_id": ["t1", "t2", "t3"],
        }
    )
    client = CapturingClient()

    _, metadata = run_pipeline(
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
        relationship="auto",
        string_exact_match_prune="to",
        reason=False,
        model="gemini-3.1-flash-lite-preview",
        batch_size=10,
        max_candidates=5,
        output_dir=tmp_path / "outputs_prune_to",
        llm_client=client,
        output_write_parquet=False,
    )

    assert len(client.payloads) == 1
    payload = client.payloads[0]
    assert [item["from_name"] for item in payload["items"]] == ["Delhi Rural"]
    assert [candidate["to_name"] for candidate in payload["items"][0]["candidates"]] == [
        "delhi extension",
    ]
    assert metadata["request"]["string_exact_match_prune"] == "to"


def test_string_exact_match_prune_none_leaves_all_rows_for_ai(tmp_path: Path):
    df_from = pd.DataFrame(
        {
            "state": ["STATE A", "STATE A"],
            "district": ["DELHI", "Delhi Rural"],
            "unit_id": ["f1", "f2"],
        }
    )
    df_to = pd.DataFrame(
        {
            "state": ["state a", "state a"],
            "district": ["delhi", "delhi extension"],
            "unit_id": ["t1", "t2"],
        }
    )
    client = CapturingClient()

    run_pipeline(
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
        relationship="auto",
        string_exact_match_prune="none",
        reason=False,
        model="gemini-3.1-flash-lite-preview",
        batch_size=10,
        max_candidates=5,
        output_dir=tmp_path / "outputs_prune_none",
        llm_client=client,
        output_write_parquet=False,
    )

    assert len(client.payloads) == 2
    assert [payload["items"][0]["from_name"] for payload in client.payloads] == [
        "DELHI",
        "Delhi Rural",
    ]
