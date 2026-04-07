from __future__ import annotations

from pathlib import Path

from adminlineage.io import read_jsonl
from adminlineage.llm import MockClient, TransientLLMError
from adminlineage.llm.base import BaseLLMClient
from adminlineage.pipeline import run_pipeline


def test_resume_skips_completed_from_units(sample_df_from, sample_df_to, tmp_path: Path):
    client = MockClient()
    output_dir = tmp_path / "outputs"

    run_kwargs = dict(
        country="India",
        year_from=1951,
        year_to=2001,
        map_col_from="subdistrict",
        map_col_to="subdistrict",
        exact_match=["state", "district"],
        id_col_from="unit_id",
        id_col_to="unit_id",
        model="gemini-3.1-flash-lite-preview",
        batch_size=2,
        max_candidates=3,
        output_dir=output_dir,
        llm_client=client,
        output_write_parquet=False,
    )

    run_pipeline(sample_df_from, sample_df_to, **run_kwargs)
    first_calls = client.calls
    assert first_calls > 0

    run_pipeline(sample_df_from, sample_df_to, **run_kwargs)
    second_calls = client.calls
    assert second_calls == first_calls

    run_dir = output_dir / "india_1951_2001_subdistrict"
    assert (run_dir / "links_raw.jsonl").exists()


def test_resume_archives_stale_raw_links_when_request_changes(
    sample_df_from,
    sample_df_to,
    tmp_path: Path,
):
    client = MockClient()
    output_dir = tmp_path / "outputs"

    common_kwargs = dict(
        country="India",
        year_from=1951,
        year_to=2001,
        map_col_from="subdistrict",
        map_col_to="subdistrict",
        exact_match=["state", "district"],
        id_col_from="unit_id",
        id_col_to="unit_id",
        model="gemini-3.1-flash-lite-preview",
        batch_size=2,
        max_candidates=3,
        output_dir=output_dir,
        llm_client=client,
        output_write_parquet=False,
    )

    run_pipeline(sample_df_from, sample_df_to, relationship="auto", **common_kwargs)
    first_calls = client.calls

    crosswalk, metadata = run_pipeline(
        sample_df_from,
        sample_df_to,
        relationship="father_to_child",
        **common_kwargs,
    )

    assert client.calls > first_calls
    assert set(crosswalk["relationship"]) == {"father_to_child"}

    run_dir = output_dir / "india_1951_2001_subdistrict"
    archived_files = list(run_dir.glob("links_raw.archive-*.jsonl"))
    assert archived_files

    active_records = read_jsonl(run_dir / "links_raw.jsonl")
    active_run_ids = {record["run_id"] for record in active_records if "run_id" in record}
    assert active_run_ids == {metadata["run_id"]}


class OneRowFailsOnceClient(BaseLLMClient):
    def __init__(self) -> None:
        self.failed_counts: dict[str, int] = {}
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
        import json

        _ = (model, temperature, seed, enable_google_search)
        self.calls += 1
        marker = "INPUT_PAYLOAD_JSON:\n"
        payload = json.loads(prompt.split(marker, maxsplit=1)[1].strip())
        decisions = []
        for item in payload["items"]:
            if item["from_key"] == "from_0":
                self.failed_counts[item["from_key"]] = (
                    self.failed_counts.get(item["from_key"], 0) + 1
                )
            if item["from_key"] == "from_0" and self.failed_counts[item["from_key"]] <= 1:
                raise TransientLLMError("temporary disconnect")

            candidate = item["candidates"][0]
            decisions.append(
                {
                    "from_key": item["from_key"],
                    "links": [
                        {
                            "to_key": candidate["to_key"],
                            "link_type": "rename",
                            "relationship": "father_to_father",
                            "score": 0.9,
                            "evidence": "Recovered on retry.",
                        }
                    ],
                }
            )
        return {"decisions": decisions}


def test_resume_retries_rows_that_only_have_error_records(
    sample_df_from,
    sample_df_to,
    tmp_path: Path,
):
    output_dir = tmp_path / "outputs_partial_resume"

    first_client = OneRowFailsOnceClient()
    run_kwargs = dict(
        country="India",
        year_from=1951,
        year_to=2001,
        map_col_from="subdistrict",
        map_col_to="subdistrict",
        exact_match=["state", "district"],
        id_col_from="unit_id",
        id_col_to="unit_id",
        model="gemini-3.1-flash-lite-preview",
        batch_size=2,
        max_candidates=3,
        output_dir=output_dir,
        output_write_parquet=False,
    )

    first_crosswalk, first_metadata = run_pipeline(
        sample_df_from,
        sample_df_to,
        llm_client=first_client,
        **run_kwargs,
    )
    assert first_metadata["counts"]["error_from_rows"] == 1
    assert first_crosswalk["evidence"].str.contains("Adjudication failed after retries").any()

    first_run_calls = first_client.calls
    second_crosswalk, second_metadata = run_pipeline(
        sample_df_from,
        sample_df_to,
        llm_client=first_client,
        **run_kwargs,
    )

    assert second_metadata["counts"]["error_from_rows"] == 0
    assert not second_crosswalk["evidence"].str.contains(
        "Adjudication failed after retries"
    ).any()
    assert first_client.calls == first_run_calls + 1
