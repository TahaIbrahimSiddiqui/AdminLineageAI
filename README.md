# adminlineage

`adminlineage` builds administrative evolution keys (crosswalks) between two time periods for applied research workflows in pandas.

It is designed for cases like mapping `subdistrict` units from period A to period B while constraining search with exact-match anchors such as `state` and `district`.

## What It Does

Given `df_from` and `df_to`, the package:

1. Normalizes names.
2. Generates cheap lexical candidate shortlists inside anchor groups.
3. Uses Gemini to adjudicate ambiguous links (`rename`, `split`, `merge`, `transfer`, `no_match`, `unknown`).
4. Runs global consistency checks.
5. Writes resumable outputs and metadata.

The model uses place names and hierarchical context supplied by the user. It does not require narrative historical documents.

## Install (local repo)

```bash
python -m pip install -e .[dev]
```

## Quickstart (pandas API)

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
    map_col_to="subdistrict",
    anchor_cols=["state", "district"],
    id_col_from="unit_id",
    id_col_to="unit_id",
    aliases=aliases,
    model="gemini-2.5-pro",
    resume_dir="outputs",
    run_name="quickstart",
)

print(crosswalk_df.head())
print(metadata["artifacts"])
```

## CLI

```bash
adminlineage preview --config examples/config/example.yml
adminlineage validate --config examples/config/example.yml
adminlineage run --config examples/config/example.yml
adminlineage export --input outputs/example_run/evolution_key.parquet --format csv
```

## Config Example

See [`examples/config/example.yml`](examples/config/example.yml).

Required fields:

- `request.country`
- `request.year_from`, `request.year_to`
- `request.map_col_from` (`map_col_to` defaults to the same value)

Optional but recommended:

- `request.anchor_cols` for exact-match constraints (for example `state`, `district`)
- `data.aliases_path` for known rename seeds

## How Anchors Work

If `anchor_cols` are provided, `df_from` and `df_to` are only compared within identical anchor tuples.

Example: with `anchor_cols=["state", "district"]`, a row from `(S1, D1)` is only matched against candidates in `(S1, D1)`.

If anchors are omitted, the tool runs in one global group and emits a warning because false positives are more likely.

## Output Artifacts

For each run directory (for example `outputs/example_run`):

- `links_raw.jsonl` (incremental/resumable batch records)
- `evolution_key.csv`
- `evolution_key.parquet` (if parquet engine available)
- `review_queue.csv`
- `run_metadata.json`

Crosswalk columns include:

- `from_name`, `to_name`
- `from_canonical_name`, `to_canonical_name`
- `from_id`, `to_id`
- `score`, `link_type`, `evidence`
- anchor columns
- `country`, `year_from`, `year_to`, `run_id`

## Safety and Validation Caveats

- Treat outputs as model-assisted mappings, not ground truth.
- Always review `review_queue.csv` before publication or policy use.
- Spot-check against trusted lists, gazetteers, or known administrative changes.
- Keep evidence short and auditable; chain-of-thought is not requested or stored.

## Documentation

- [`docs/architecture.md`](docs/architecture.md)
- [`docs/usage.md`](docs/usage.md)

## Development

```bash
ruff check .
python -m pytest
```
