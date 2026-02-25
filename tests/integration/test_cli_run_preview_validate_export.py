from __future__ import annotations

from pathlib import Path

import pandas as pd

from adminlineage.cli import main


def _write_example_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    df_from = pd.DataFrame(
        {
            "state": ["S1"],
            "district": ["D1"],
            "subdistrict": ["North Block"],
            "unit_id": ["f1"],
        }
    )
    df_to = pd.DataFrame(
        {
            "state": ["S1"],
            "district": ["D1"],
            "subdistrict": ["North Block"],
            "unit_id": ["t1"],
        }
    )
    from_path = tmp_path / "from.csv"
    to_path = tmp_path / "to.csv"
    df_from.to_csv(from_path, index=False)
    df_to.to_csv(to_path, index=False)

    config_path = tmp_path / "config.yml"
    config_path.write_text(
        f"""
request:
  country: India
  year_from: 1951
  year_to: 2001
  map_col_from: subdistrict
  map_col_to: subdistrict
  anchor_cols: [state, district]
  id_col_from: unit_id
  id_col_to: unit_id
data:
  mode: files
  from_path: {from_path}
  to_path: {to_path}
llm:
  provider: mock
  model: gemini-2.5-pro
pipeline:
  batch_size: 10
  max_candidates: 5
  run_name: cli_run
output:
  directory: {tmp_path / 'outputs'}
  write_parquet: false
""".strip(),
        encoding="utf-8",
    )
    return from_path, to_path, config_path


def test_cli_run_preview_validate_and_export(tmp_path: Path):
    _, _, config_path = _write_example_files(tmp_path)

    assert main(["preview", "--config", str(config_path)]) == 0
    assert main(["validate", "--config", str(config_path)]) == 0
    assert main(["run", "--config", str(config_path)]) == 0

    run_dir = tmp_path / "outputs" / "cli_run"
    csv_path = run_dir / "evolution_key.csv"
    assert csv_path.exists()

    output_jsonl = run_dir / "evolution_key_export.jsonl"
    assert (
        main(
            [
                "export",
                "--input",
                str(csv_path),
                "--format",
                "jsonl",
                "--output",
                str(output_jsonl),
            ]
        )
        == 0
    )
    assert output_jsonl.exists()
