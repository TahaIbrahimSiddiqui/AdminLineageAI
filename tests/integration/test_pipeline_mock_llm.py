from __future__ import annotations

from pathlib import Path

from adminlineage.llm import MockClient
from adminlineage.pipeline import run_pipeline


def test_pipeline_end_to_end_with_mock(sample_df_from, sample_df_to, sample_aliases, tmp_path: Path):
    client = MockClient(default_score=0.91)

    crosswalk, metadata = run_pipeline(
        sample_df_from,
        sample_df_to,
        country="India",
        year_from=1951,
        year_to=2001,
        map_col_from="subdistrict",
        map_col_to="subdistrict",
        anchor_cols=["state", "district"],
        id_col_from="unit_id",
        id_col_to="unit_id",
        aliases=sample_aliases,
        model="gemini-2.5-pro",
        batch_size=2,
        max_candidates=3,
        resume_dir=tmp_path,
        run_name="integration_run",
        llm_client=client,
        output_write_parquet=False,
    )

    assert not crosswalk.empty
    assert {"from_name", "to_name", "score", "link_type", "run_id"}.issubset(crosswalk.columns)
    assert metadata["run_id"]
    run_dir = tmp_path / "integration_run"
    assert (run_dir / "evolution_key.csv").exists()
    assert (run_dir / "review_queue.csv").exists()
    assert (run_dir / "run_metadata.json").exists()
    assert (run_dir / "links_raw.jsonl").exists()
