from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from pydantic import BaseModel

from adminlineage.llm.base import BaseLLMClient, QuotaExceededLLMError, TransientLLMError
from adminlineage.pipeline import run_pipeline
from adminlineage.schema import normalize_nullable_output_columns


class SplitOnLargeBatchClient(BaseLLMClient):
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

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
        marker = "INPUT_PAYLOAD_JSON:\n"
        payload = json.loads(prompt.split(marker, maxsplit=1)[1].strip())
        items = payload["items"]
        include_evidence = bool(payload.get("include_evidence", False))
        self.batch_sizes.append(len(items))

        if len(items) > 1:
            raise RuntimeError("simulated batch failure")

        item = items[0]
        candidate = item["candidates"][0]
        response = {
            "decisions": [
                {
                    "from_key": item["from_key"],
                    "links": [
                        {
                            "to_key": candidate["to_key"],
                            "link_type": "rename",
                            "relationship": "father_to_father",
                            "score": 0.95,
                        }
                    ],
                }
            ]
        }
        if include_evidence:
            response["decisions"][0]["links"][0]["evidence"] = "Recovered after splitting batch."
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            return schema.model_validate(response).model_dump()
        return response


def test_pipeline_runs_in_single_row_mode_without_batch_split_retries(
    sample_df_from,
    sample_df_to,
    tmp_path: Path,
):
    client = SplitOnLargeBatchClient()

    crosswalk, metadata = run_pipeline(
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
        evidence=True,
        reason=False,
        model="gemini-3.1-flash-lite-preview",
        batch_size=2,
        max_candidates=3,
        output_dir=tmp_path / "outputs_batch_split",
        llm_client=client,
        output_write_parquet=False,
    )

    assert client.batch_sizes == [1, 1, 1]
    assert crosswalk["link_type"].eq("rename").all()
    assert not any("No completed adjudication record." in text for text in crosswalk["evidence"])
    assert any(
        "Using effective batch_size=1 instead." in warning for warning in metadata["warnings"]
    )


class FlakySingleRowClient(BaseLLMClient):
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []
        self.failed_from_keys: list[str] = []

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
        marker = "INPUT_PAYLOAD_JSON:\n"
        payload = json.loads(prompt.split(marker, maxsplit=1)[1].strip())
        items = payload["items"]
        include_evidence = bool(payload.get("include_evidence", False))
        self.batch_sizes.append(len(items))

        if len(items) > 1:
            raise TransientLLMError("temporary provider failure")

        item = items[0]
        if item["from_key"] == "from_0":
            self.failed_from_keys.append(item["from_key"])
            raise TransientLLMError("server disconnected without sending a response")

        candidate = item["candidates"][0]
        response = {
            "decisions": [
                {
                    "from_key": item["from_key"],
                    "links": [
                        {
                            "to_key": candidate["to_key"],
                            "link_type": "rename",
                            "relationship": "father_to_father",
                            "score": 0.95,
                        }
                    ],
                }
            ]
        }
        if include_evidence:
            response["decisions"][0]["links"][0]["evidence"] = "Recovered after splitting batch."
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            return schema.model_validate(response).model_dump()
        return response


def test_pipeline_surfaces_unrecovered_row_failures_after_transient_retries(
    sample_df_from,
    sample_df_to,
    tmp_path: Path,
):
    client = FlakySingleRowClient()

    crosswalk, metadata = run_pipeline(
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
        evidence=True,
        reason=False,
        model="gemini-3.1-flash-lite-preview",
        batch_size=2,
        max_candidates=3,
        output_dir=tmp_path / "outputs_transient_row_failure",
        llm_client=client,
        output_write_parquet=False,
    )

    assert client.batch_sizes == [1, 1, 1]
    assert client.failed_from_keys == ["from_0"]
    assert metadata["counts"]["error_from_rows"] == 1
    assert any(
        "Adjudication still failed for 1 source rows after retries." in warning
        for warning in metadata["warnings"]
    )

    failed_row = crosswalk.loc[crosswalk["from_key"] == "from_0"].iloc[0]
    assert failed_row["to_key"] is None
    assert "Adjudication failed after retries" in failed_row["evidence"]
    assert "server disconnected without sending a response" in failed_row["evidence"]


def test_normalize_nullable_crosswalk_columns_converts_missing_targets_to_none():
    crosswalk = pd.DataFrame(
        [
            {
                "from_key": "from_0",
                "to_key": "to_0",
                "to_name": "North Block",
                "to_canonical_name": "north block",
                "to_id": "t1",
            },
            {
                "from_key": "from_1",
                "to_key": float("nan"),
                "to_name": pd.NA,
                "to_canonical_name": None,
                "to_id": float("nan"),
            },
        ]
    )

    normalized = normalize_nullable_output_columns(crosswalk)
    missing_row = normalized.loc[normalized["from_key"] == "from_1"].iloc[0]

    assert missing_row["to_key"] is None
    assert missing_row["to_name"] is None
    assert missing_row["to_canonical_name"] is None
    assert missing_row["to_id"] is None


class SpendingCapClient(BaseLLMClient):
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
        _ = (prompt, schema, model, temperature, seed, enable_google_search)
        raise QuotaExceededLLMError("Gemini spending cap reached.")


def test_pipeline_raises_spending_cap_errors_without_batch_fallback(tmp_path: Path):
    df_from = pd.DataFrame(
        [
            {"subdistrict": "North Block", "unit_id": "f1"},
            {"subdistrict": "River Tehsil", "unit_id": "f2"},
        ]
    )
    df_to = pd.DataFrame(
        [
            {"subdistrict": "North Block New", "unit_id": "t1"},
            {"subdistrict": "River Tehsil New", "unit_id": "t2"},
        ]
    )

    with pytest.raises(QuotaExceededLLMError, match="spending cap reached"):
        run_pipeline(
            df_from=df_from,
            df_to=df_to,
            country="India",
            year_from=1951,
            year_to=2001,
            map_col_from="subdistrict",
            map_col_to="subdistrict",
            id_col_from="unit_id",
            id_col_to="unit_id",
            relationship="auto",
            reason=False,
            model="gemini-3.1-flash-lite-preview",
            batch_size=2,
            max_candidates=3,
            output_dir=tmp_path / "outputs_spending_cap",
            llm_client=SpendingCapClient(),
            output_write_parquet=False,
        )
