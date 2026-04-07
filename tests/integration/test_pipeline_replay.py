from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from adminlineage.io import read_jsonl
from adminlineage.llm import MockClient
from adminlineage.llm.base import BaseLLMClient
from adminlineage.pipeline import run_pipeline


def _run_dir(base_dir: Path) -> Path:
    return base_dir / "india_1951_2001_subdistrict"


class GroundedReplayClient(BaseLLMClient):
    def __init__(self) -> None:
        self.calls = 0

    def generate_json(
        self,
        prompt,
        schema,
        model,
        temperature,
        seed,
        *,
        enable_google_search: bool = False,
    ):
        _ = (model, temperature, seed, enable_google_search)
        self.calls += 1
        payload = json.loads(prompt.split("INPUT_PAYLOAD_JSON:\n", maxsplit=1)[1].strip())
        include_evidence = bool(payload.get("include_evidence", False))
        item = payload["items"][0]
        if "grounding_context" in item:
            link = {
                "to_key": "to_0",
                "link_type": "rename",
                "relationship": "father_to_father",
                "score": 0.92,
            }
        else:
            link = {
                "to_key": None,
                "link_type": "unknown",
                "relationship": "unknown",
                "score": 0.35,
            }
        if include_evidence:
            link["evidence"] = (
                "Grounded replay test match."
                if "grounding_context" in item
                else "Needs grounded verification."
            )
        response = {
            "decisions": [
                {
                    "from_key": "from_0",
                    "links": [link],
                }
            ]
        }
        if isinstance(schema, type):
            return schema.model_validate(response).model_dump()
        return response

    def generate_text(
        self,
        prompt,
        model,
        temperature,
        seed,
        *,
        enable_google_search: bool = False,
    ) -> str:
        _ = (prompt, model, temperature, seed, enable_google_search)
        self.calls += 1
        return "to_0 is supported by grounded verification."


def test_pipeline_replay_hit_skips_llm_and_materializes_outputs(
    sample_df_from,
    sample_df_to,
    tmp_path: Path,
):
    replay_store_dir = tmp_path / "shared-replay"
    first_output_dir = tmp_path / "outputs-live"
    second_output_dir = tmp_path / "outputs-replay"

    first_client = MockClient(default_score=0.91)
    first_crosswalk, first_metadata = run_pipeline(
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
        output_dir=first_output_dir,
        llm_client=first_client,
        temperature=0.75,
        enable_google_search=True,
        cache_path="first-cache.sqlite",
        replay_enabled=True,
        replay_store_dir=replay_store_dir,
        env_search_dir=tmp_path / "env-one",
        output_write_parquet=False,
    )

    assert first_client.calls > 0
    assert first_metadata["execution_mode"] == "live"
    assert first_metadata["replay_hit"] is False
    assert first_metadata["replay_key"]

    second_client = MockClient(default_score=0.91)
    second_crosswalk, second_metadata = run_pipeline(
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
        output_dir=second_output_dir,
        llm_client=second_client,
        temperature=0.75,
        enable_google_search=True,
        cache_path="second-cache.sqlite",
        replay_enabled=True,
        replay_store_dir=replay_store_dir,
        env_search_dir=tmp_path / "env-two",
        output_write_parquet=False,
    )

    assert second_client.calls == 0
    assert second_metadata["execution_mode"] == "replay"
    assert second_metadata["replay_hit"] is True
    assert second_metadata["replay_key"] == first_metadata["replay_key"]
    assert second_metadata["replayed_from_run_id"] == first_metadata["run_id"]
    pd.testing.assert_frame_equal(first_crosswalk, second_crosswalk)

    run_dir = _run_dir(second_output_dir)
    assert (run_dir / "evolution_key.csv").exists()
    assert (run_dir / "review_queue.csv").exists()
    assert (run_dir / "run_metadata.json").exists()
    assert read_jsonl(run_dir / "links_raw.jsonl")


def test_pipeline_replay_miss_runs_live_when_semantic_request_changes(
    sample_df_from,
    sample_df_to,
    tmp_path: Path,
):
    replay_store_dir = tmp_path / "shared-replay"
    first_client = MockClient(default_score=0.91)
    _, first_metadata = run_pipeline(
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
        output_dir=tmp_path / "outputs-first",
        llm_client=first_client,
        replay_enabled=True,
        replay_store_dir=replay_store_dir,
        output_write_parquet=False,
    )

    second_client = MockClient(default_score=0.91)
    crosswalk, second_metadata = run_pipeline(
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
        relationship="father_to_child",
        reason=False,
        model="gemini-3.1-flash-lite-preview",
        batch_size=2,
        max_candidates=3,
        output_dir=tmp_path / "outputs-second",
        llm_client=second_client,
        replay_enabled=True,
        replay_store_dir=replay_store_dir,
        output_write_parquet=False,
    )

    assert second_client.calls > 0
    assert second_metadata["execution_mode"] == "live"
    assert second_metadata["replay_hit"] is False
    assert second_metadata["replay_key"] != first_metadata["replay_key"]
    assert set(crosswalk["relationship"]) == {"father_to_child"}


def test_pipeline_replay_corruption_falls_back_to_live(
    sample_df_from,
    sample_df_to,
    tmp_path: Path,
):
    replay_store_dir = tmp_path / "shared-replay"
    first_client = MockClient(default_score=0.91)
    _, first_metadata = run_pipeline(
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
        output_dir=tmp_path / "outputs-first",
        llm_client=first_client,
        replay_enabled=True,
        replay_store_dir=replay_store_dir,
        output_write_parquet=False,
    )

    replay_dir = replay_store_dir / first_metadata["replay_key"]
    (replay_dir / "crosswalk.records.json").write_text("{not valid json", encoding="utf-8")

    second_client = MockClient(default_score=0.91)
    _, second_metadata = run_pipeline(
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
        output_dir=tmp_path / "outputs-second",
        llm_client=second_client,
        replay_enabled=True,
        replay_store_dir=replay_store_dir,
        output_write_parquet=False,
    )

    assert second_client.calls > 0
    assert second_metadata["execution_mode"] == "live"
    assert second_metadata["replay_hit"] is False
    assert any("Replay bundle load failed" in warning for warning in second_metadata["warnings"])


def test_pipeline_replay_round_trips_grounding_notes(tmp_path: Path):
    df_from = pd.DataFrame({"subdistrict": ["Beta District"], "unit_id": ["f1"]})
    df_to = pd.DataFrame({"subdistrict": ["Beta District New"], "unit_id": ["t1"]})
    replay_store_dir = tmp_path / "shared-replay"

    live_client = GroundedReplayClient()
    _, live_metadata = run_pipeline(
        df_from,
        df_to,
        country="India",
        year_from=1951,
        year_to=2001,
        map_col_from="subdistrict",
        map_col_to="subdistrict",
        exact_match=[],
        id_col_from="unit_id",
        id_col_to="unit_id",
        relationship="auto",
        reason=False,
        model="gemini-3.1-flash-lite-preview",
        batch_size=1,
        max_candidates=1,
        output_dir=tmp_path / "outputs-grounded-live",
        llm_client=live_client,
        temperature=0.75,
        enable_google_search=True,
        replay_enabled=True,
        replay_store_dir=replay_store_dir,
        output_write_parquet=False,
    )

    live_run_dir = (tmp_path / "outputs-grounded-live") / "india_1951_2001_subdistrict"
    assert (live_run_dir / "grounding_notes.jsonl").exists()

    replay_client = GroundedReplayClient()
    _, replay_metadata = run_pipeline(
        df_from,
        df_to,
        country="India",
        year_from=1951,
        year_to=2001,
        map_col_from="subdistrict",
        map_col_to="subdistrict",
        exact_match=[],
        id_col_from="unit_id",
        id_col_to="unit_id",
        relationship="auto",
        reason=False,
        model="gemini-3.1-flash-lite-preview",
        batch_size=1,
        max_candidates=1,
        output_dir=tmp_path / "outputs-grounded-replay",
        llm_client=replay_client,
        temperature=0.75,
        enable_google_search=True,
        replay_enabled=True,
        replay_store_dir=replay_store_dir,
        output_write_parquet=False,
    )

    replay_run_dir = (tmp_path / "outputs-grounded-replay") / "india_1951_2001_subdistrict"
    assert replay_client.calls == 0
    assert replay_metadata["execution_mode"] == "replay"
    assert (replay_run_dir / "grounding_notes.jsonl").exists()
    assert read_jsonl(replay_run_dir / "grounding_notes.jsonl")
