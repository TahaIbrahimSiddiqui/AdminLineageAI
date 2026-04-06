from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel

from adminlineage.llm.base import BaseLLMClient
from adminlineage.pipeline import run_pipeline


class CapturingClient(BaseLLMClient):
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


def test_pipeline_keeps_search_enabled_without_prompt_grounding(
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
        model="gemini-2.5-flash",
        batch_size=2,
        max_candidates=3,
        output_dir=tmp_path / "outputs_prompt_behavior",
        llm_client=client,
        temperature=1.0,
        enable_google_search=True,
        output_write_parquet=False,
    )

    assert client.search_flags == [True, True]
    assert client.prompts
    assert all(
        "Use only the provided names and hierarchical context." in prompt
        for prompt in client.prompts
    )
    assert all("grounding tools are available" not in prompt for prompt in client.prompts)


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
        model="gemini-2.5-flash",
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
        model="gemini-2.5-flash",
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
        model="gemini-2.5-flash",
        batch_size=10,
        max_candidates=5,
        output_dir=tmp_path / "outputs_prune_none",
        llm_client=client,
        output_write_parquet=False,
    )

    assert len(client.payloads) == 1
    payload = client.payloads[0]
    assert [item["from_name"] for item in payload["items"]] == ["DELHI", "Delhi Rural"]
