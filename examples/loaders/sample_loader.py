"""Example python_hook loader for CLI config."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_data(config: dict):
    """Return (df_from, df_to, aliases) for adminlineage python_hook mode."""

    base = Path(config.get("base_dir", "examples/data"))
    df_from = pd.read_csv(base / "from_units.csv")
    df_to = pd.read_csv(base / "to_units.csv")
    aliases = pd.read_csv(base / "aliases.csv")
    return df_from, df_to, aliases
