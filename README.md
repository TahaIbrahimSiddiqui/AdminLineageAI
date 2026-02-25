# adminlineage

`adminlineage` builds administrative evolution keys (crosswalks) between two time periods for pandas workflows.

## What is the evolution key?

An evolution key is a table of links from period A units (`df_from`) to period B units (`df_to`).

- One `from` unit can map to multiple `to` units (`split`).
- Multiple `from` units can map to one `to` unit (`merge`).
- Links are constrained by exact anchor columns when anchors are provided.

Each row in `evolution_key.csv` is one proposed lineage link with score, link type, evidence, and context.

## Install (local repo)

```bash
python -m pip install -e .[dev,io]
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
    gemini_api_key_env="GEMINI_API_KEY",
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

## Required vs optional inputs

### Required API arguments

| Argument | Required | Meaning |
|---|---|---|
| `df_from` | yes | DataFrame for period A units |
| `df_to` | yes | DataFrame for period B units |
| `country` | yes | Country label included in prompts and metadata |
| `year_from` | yes | Label for period A |
| `year_to` | yes | Label for period B |
| `map_col_from` | yes | Name column in `df_from` to map |

### Optional API arguments

| Argument | Required | Meaning |
|---|---|---|
| `map_col_to` | no | Name column in `df_to`; defaults to `map_col_from` |
| `anchor_cols` | no | Exact-match parent columns (for example `state`, `district`) |
| `id_col_from` | no | Stable ID column in `df_from`; falls back to internal key |
| `id_col_to` | no | Stable ID column in `df_to`; falls back to internal key |
| `extra_context_cols` | no | Extra columns included in LLM context |
| `aliases` | no | Seed rename table (`from_alias`, `to_alias`, optional anchors) |
| `model` | no | Gemini model name; default `gemini-2.5-pro` |
| `gemini_api_key_env` | no | Env var name containing API key |
| `batch_size` | no | LLM adjudication batch size |
| `max_candidates` | no | Candidate shortlist cap per `from` unit |
| `resume_dir` | no | Base output/resume folder |
| `run_name` | no | Subfolder name under `resume_dir` |
| `seed` | no | Seed for deterministic behavior |

## Required vs optional DataFrame columns

### `df_from`

| Column type | Required | Notes |
|---|---|---|
| Mapping name column (`map_col_from`) | yes | Core unit name being mapped |
| Anchor columns (`anchor_cols`) | conditional | Required only if `anchor_cols` is provided |
| ID column (`id_col_from`) | no | Optional stable identifier |
| Extra context columns | no | Optional prompt context |

### `df_to`

| Column type | Required | Notes |
|---|---|---|
| Mapping name column (`map_col_to` or `map_col_from`) | yes | Target unit name |
| Anchor columns (`anchor_cols`) | conditional | Required only if `anchor_cols` is provided |
| ID column (`id_col_to`) | no | Optional stable identifier |
| Extra context columns | no | Optional prompt context |

### `aliases` DataFrame (optional)

| Column | Required | Meaning |
|---|---|---|
| `from_alias` | yes | Known historical/alternate name in period A |
| `to_alias` | yes | Known target name in period B |
| Anchor columns | no | If present, alias is scoped to those anchors |

## `resume_dir` and `run_name`

- Effective run folder is `resume_dir/run_name`.
- `links_raw.jsonl` is written incrementally and used for resume.
- Re-running with the same `resume_dir` and `run_name` skips completed `from` units.
- Use a new `run_name` for a clean run.

## Gemini API key setup

Set environment variable before running:

```bash
export GEMINI_API_KEY="your_api_key"
```

CLI config must use Gemini provider:

```yaml
llm:
  provider: gemini
  model: gemini-2.5-pro
  gemini_api_key_env: GEMINI_API_KEY
```

Custom env var names are supported:

```bash
export MY_GEMINI_KEY="your_api_key"
```

```yaml
llm:
  provider: gemini
  gemini_api_key_env: MY_GEMINI_KEY
```

## Output artifacts

For each run directory (for example `outputs/example_run`):

- `links_raw.jsonl` incremental records for resumability
- `evolution_key.csv` final crosswalk
- `evolution_key.parquet` final crosswalk (if parquet engine installed)
- `review_queue.csv` links that need manual review
- `run_metadata.json` request/settings/counts/artifact paths

## Crosswalk columns (what each means)

| Column | Meaning |
|---|---|
| `from_name` | Raw source name from `df_from` |
| `to_name` | Raw target name from `df_to` (can be null for no-match) |
| `from_canonical_name` | Normalized source name used for matching |
| `to_canonical_name` | Normalized target name used for matching |
| `from_id` | Source ID from `id_col_from` or internal fallback |
| `to_id` | Target ID from `id_col_to` or internal fallback |
| `score` | Model confidence in `[0,1]` |
| `link_type` | One of `rename`, `split`, `merge`, `transfer`, `no_match`, `unknown` |
| `evidence` | Short model summary, no chain-of-thought |
| `country` | Request country |
| `year_from` | Request period A label |
| `year_to` | Request period B label |
| `run_id` | Deterministic run identifier |
| `from_key` | Internal source unit key |
| `to_key` | Internal target unit key (nullable) |
| `constraints_passed` | JSON object with checks like `candidate_membership` and `anchor_match` |
| `<anchor columns>` | Any user-provided anchor fields carried into output |
| `review_flags` | List of issues used for review queue |
| `review_reason` | Comma-separated summary of review flags |

## How anchors work

If `anchor_cols` are provided, `df_from` and `df_to` are only compared within identical anchor tuples.

Example: with `anchor_cols=["state", "district"]`, a row from `(S1, D1)` is only matched against candidates in `(S1, D1)`.

If anchors are omitted, the tool runs in one global group and emits a warning because false positives are more likely.

## Safety and validation caveats

- Treat outputs as model-assisted mappings, not ground truth.
- Always review `review_queue.csv` before publication or policy use.
- Spot-check against trusted lists, gazetteers, or known administrative changes.
- Evidence is intentionally short and auditable; chain-of-thought is not requested or stored.

## Documentation

- [`docs/architecture.md`](docs/architecture.md)
- [`docs/usage.md`](docs/usage.md)
- [`docs/output_schema.json`](docs/output_schema.json)

## Development

```bash
ruff check .
python -m pytest
```
