from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from adminlineage.config import load_config


def test_load_config_files_mode(tmp_path: Path):
    cfg_path = tmp_path / "config.yml"
    cfg_path.write_text(
        """
request:
  country: India
  year_from: 1951
  year_to: 2001
  map_col_from: subdistrict
  exact_match: [state, district]
  string_exact_match_prune: to
  relationship: father_to_child
  reason: true
data:
  mode: files
  from_path: examples/data/from_units.csv
  to_path: examples/data/to_units.csv
llm:
  provider: mock
  temperature: 0.75
  enable_google_search: true
replay:
  enabled: true
  store_dir: shared-replay
pipeline:
  batch_size: 5
  max_candidates: 10
output:
  write_parquet: false
""".strip(),
        encoding="utf-8",
    )

    cfg = load_config(cfg_path)
    assert cfg.request.country == "India"
    assert cfg.data.mode == "files"
    assert cfg.request.exact_match == ["state", "district"]
    assert cfg.request.string_exact_match_prune == "to"
    assert cfg.request.relationship == "father_to_child"
    assert cfg.request.reason is True
    assert cfg.llm.temperature == 0.75
    assert cfg.llm.enable_google_search is True
    assert cfg.replay.enabled is True
    assert cfg.replay.store_dir == "shared-replay"
    assert cfg.pipeline.max_candidates == 10
    assert cfg.source_dir == tmp_path.resolve()


def test_load_config_uses_default_max_candidates_of_six(tmp_path: Path):
    cfg_path = tmp_path / "default-max-candidates.yml"
    cfg_path.write_text(
        """
request:
  country: India
  year_from: 1951
  year_to: 2001
  map_col_from: subdistrict
data:
  mode: files
  from_path: examples/data/from_units.csv
  to_path: examples/data/to_units.csv
llm:
  provider: mock
""".strip(),
        encoding="utf-8",
    )

    cfg = load_config(cfg_path)
    assert cfg.pipeline.max_candidates == 6


def test_load_config_invalid_files_mode(tmp_path: Path):
    cfg_path = tmp_path / "bad.yml"
    cfg_path.write_text(
        """
request:
  country: India
  year_from: 1951
  year_to: 2001
  map_col_from: subdistrict
data:
  mode: files
  from_path: only_one.csv
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_config(cfg_path)


def test_load_config_rejects_removed_fields(tmp_path: Path):
    cfg_path = tmp_path / "removed.yml"
    cfg_path.write_text(
        """
request:
  country: India
  year_from: 1951
  year_to: 2001
  map_col_from: subdistrict
  anchor_cols: [state, district]
data:
  mode: files
  from_path: from.csv
  to_path: to.csv
  aliases_path: aliases.csv
output:
  directory: outputs
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_config(cfg_path)
