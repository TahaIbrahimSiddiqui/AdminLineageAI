# Usage

## Notebook API

```python
import pandas as pd
import adminlineage

from_df = pd.read_csv("examples/data/from_units.csv")
to_df = pd.read_csv("examples/data/to_units.csv")
aliases = pd.read_csv("examples/data/aliases.csv")

crosswalk_df, metadata = adminlineage.build_evolution_key(
    from_df,
    to_df,
    country="India",
    year_from=1951,
    year_to=2001,
    map_col_from="subdistrict",
    anchor_cols=["state", "district"],
    id_col_from="unit_id",
    id_col_to="unit_id",
    aliases=aliases,
    model="gemini-2.5-pro",
    batch_size=25,
    max_candidates=15,
    resume_dir="outputs",
    run_name="notebook_run",
)
```

## CLI

```bash
adminlineage preview --config examples/config/example.yml
adminlineage validate --config examples/config/example.yml
adminlineage run --config examples/config/example.yml
adminlineage export --input outputs/example_run/evolution_key.parquet --format csv
```

## Config Loader Modes

- `data.mode: files` reads `from_path`, `to_path`, optional `aliases_path`.
- `data.mode: python_hook` calls `module:function(config)->(df_from, df_to[, aliases])`.

## Validation Workflow

1. Run `validate` before long jobs.
2. Run `preview` to inspect grouping and candidate budget.
3. Inspect `review_queue.csv` after run.
4. Spot-check low-score and `unknown`/`no_match` mappings before downstream use.
