from __future__ import annotations

import sys
from pathlib import Path

from adminlineage.config import RunConfig
from adminlineage.io import load_frames



def test_load_frames_python_hook(tmp_path: Path):
    module_path = tmp_path / "loader_mod.py"
    module_path.write_text(
        """
import pandas as pd

def load_pair(config):
    df_from = pd.DataFrame({"name": ["A"]})
    df_to = pd.DataFrame({"name": ["B"]})
    return df_from, df_to
""".strip(),
        encoding="utf-8",
    )
    sys.path.insert(0, str(tmp_path))
    try:
        cfg = RunConfig.model_validate(
            {
                "request": {
                    "country": "India",
                    "year_from": 1951,
                    "year_to": 2001,
                    "map_col_from": "name",
                },
                "data": {
                    "mode": "python_hook",
                    "callable": "loader_mod:load_pair",
                    "params": {"x": 1},
                },
            }
        )

        loaded = load_frames(cfg, cwd=tmp_path)
        assert loaded.df_from.shape[0] == 1
        assert loaded.loader_metadata["mode"] == "python_hook"
    finally:
        sys.path = [path for path in sys.path if path != str(tmp_path)]
