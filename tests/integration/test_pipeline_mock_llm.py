from __future__ import annotations

from pathlib import Path

import pandas as pd

from adminlineage.llm import MockClient
from adminlineage.pipeline import run_pipeline


def test_pipeline_end_to_end_with_mock(sample_df_from, sample_df_to, tmp_path: Path):
    client = MockClient(default_score=0.91)
    output_dir = tmp_path / "outputs"

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
        reason=False,
        model="gemini-2.5-pro",
        batch_size=2,
        max_candidates=3,
        output_dir=output_dir,
        llm_client=client,
        output_write_parquet=False,
    )

    assert not crosswalk.empty
    assert {
        "from_name",
        "to_name",
        "score",
        "link_type",
        "relationship",
        "reason",
        "run_id",
    }.issubset(crosswalk.columns)
    assert crosswalk["reason"].eq("").all()
    assert metadata["run_id"]
    run_dir = output_dir / "india_1951_2001_subdistrict"
    csv_path = run_dir / "evolution_key.csv"
    assert csv_path.exists()
    assert (run_dir / "review_queue.csv").exists()
    assert (run_dir / "run_metadata.json").exists()
    assert (run_dir / "links_raw.jsonl").exists()
    saved = pd.read_csv(csv_path)
    assert list(saved.columns) == list(crosswalk.columns)
    assert metadata["artifacts"]["run_metadata_json"] == str(run_dir / "run_metadata.json")


def test_pipeline_reason_mode_with_mock(sample_df_from, sample_df_to, tmp_path: Path):
    client = MockClient(default_score=0.91)

    crosswalk, _ = run_pipeline(
        sample_df_from,
        sample_df_to,
        country="India",
        year_from=1951,
        year_to=2011,
        map_col_from="subdistrict",
        map_col_to="subdistrict",
        exact_match=["state", "district"],
        id_col_from="unit_id",
        id_col_to="unit_id",
        relationship="father_to_child",
        reason=True,
        model="gemini-2.5-pro",
        batch_size=2,
        max_candidates=3,
        output_dir=tmp_path / "outputs_reason",
        llm_client=client,
        output_write_parquet=False,
    )

    assert crosswalk["reason"].str.len().gt(0).all()
    assert set(crosswalk["relationship"]) == {"father_to_child"}


def test_pipeline_writes_csv_fallback_when_parquet_fails(
    sample_df_from,
    sample_df_to,
    tmp_path: Path,
    monkeypatch,
):
    client = MockClient(default_score=0.91)

    def _raise_parquet_error(self, *args, **kwargs):
        _ = (self, args, kwargs)
        raise RuntimeError("parquet engine exploded")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", _raise_parquet_error)

    _, metadata = run_pipeline(
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
        output_dir=tmp_path / "outputs_parquet_fallback",
        llm_client=client,
        output_write_csv=False,
        output_write_parquet=True,
    )

    csv_path = (
        tmp_path
        / "outputs_parquet_fallback"
        / "india_1951_2001_subdistrict"
        / "evolution_key.csv"
    )
    assert csv_path.exists()
    assert metadata["artifacts"]["evolution_key_csv"] == str(csv_path)
    assert any("wrote CSV fallback instead" in warning for warning in metadata["warnings"])
