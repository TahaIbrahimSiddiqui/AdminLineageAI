from __future__ import annotations

from pathlib import Path

import pandas as pd

from adminlineage.io import read_jsonl
from adminlineage.llm import MockClient
from adminlineage.pipeline import run_pipeline


def _run_dir(base_dir: Path) -> Path:
    return base_dir / "india_1951_2001_subdistrict"


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
        model="gemini-2.5-pro",
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
        model="gemini-2.5-pro",
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
        model="gemini-2.5-pro",
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
        model="gemini-2.5-pro",
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
        model="gemini-2.5-pro",
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
        model="gemini-2.5-pro",
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
