# AdminLineageAI

AdminLineageAI builds crosswalks and administrative evolution keys between two datasets.

AdminLineageAI makes crosswalks between administrative locations such as districts (ADM2), subdistricts (ADM3), states (ADM1), and countries (ADM0) across datasets that may come from completely different sources and different periods. It uses AI to compare likely matches, reason over spelling variants and language-specific forms, and produce a usable crosswalk plus review artifacts.

Matching administrative units by hand is labour-intensive work. Names vary across sources, spellings shift across languages, and units are often renamed, split, or merged. Through this package, we hope to reduce the manual work of matching administrative units between datasets while still keeping a clear review trail.

You give it one table from an earlier period, one table from a later period, the name columns you want to map, and any scope columns that must agree exactly. The package generates candidate matches, asks Gemini to choose among them, and writes a crosswalk plus review artifacts. The final evolution key includes a `merge` indicator so you can tell whether a row exists on both sides, only in the earlier-period table, or only in the later-period table.

<p align="center"><sub><strong>Experimental:</strong> Treat these crosswalks as assistive outputs and cross-verify them, especially in important cases. We would love to hear about other field experiences and use cases for this package.</sub></p>

## Where It Helps

- Matching a scheme dataset from a website against a standard administrative list such as a census table. For example, one source may write `Paschimi Singhbhum` while another uses `West Singhbhum`. Plain fuzzy matching often misses cases like this unless you manually standardize prefixes and suffixes first. AI can do better because it has context, including that `paschim` in Hindi means `west`. The same kind of issue shows up across many widely spoken languages.
- Handling administrative churn. Districts and other units are regularly split, merged, renamed, or grouped differently, and there is often no up-to-date public evolution list for newly created units.
- Creating entirely new evolution crosswalks that do not already exist between two sources or two periods.

## Three Important Features

- Exact string handling plus selective pruning. Token costs can rise quickly, so the package looks for exact string hits first and lets you control later AI work with `string_exact_match_prune`. This behaviour is explained in more detail below.
- Hierarchical matching with `exact_match`. If your data are nested, you can match names within exact scopes such as `country`, `state`, or `district`. For example, you can match district names within states inside a country. This works well, but the exact-match columns need to line up exactly across both datasets.
- Replay and reproducibility. Academic pipelines often need to be rerun many times. With replay enabled, repeated semantic requests can reuse prior completed LLM work instead of calling the API again. The `seed` parameter helps keep request identity deterministic and makes reruns easier to reproduce.

The supported live workflow in AdminLineageAI is:

- Gemini `gemini-3.1-flash-lite-preview`
- Google Search grounding enabled
- strict JSON output from the model
- user-controlled batching with automatic split fallback on failed multi-row requests

## How To Use

You do not need the CLI to use AdminLineageAI. The simplest path is the Python API.

1. Install the package from the repository root.

```bash
pip install -e .
```

Install the optional parquet dependency if you want parquet output support:

```bash
pip install -e ".[io]"
```

2. Set a Gemini API key in `GEMINI_API_KEY`, or use another environment variable name and pass it explicitly.

```bash
GEMINI_API_KEY=your_api_key_here
```

The package can load a nearby `.env` file when it looks for the key.

3. Prepare two tables: one earlier-period table and one later-period table.

4. Choose the name column on each side, and add optional exact-match columns, IDs, or extra context columns if you have them.

5. Run the matcher.

```python
import pandas as pd
import adminlineage

df_from = pd.read_csv("from_units.csv")
df_to = pd.read_csv("to_units.csv")

crosswalk_df, metadata = adminlineage.build_evolution_key(
    df_from,
    df_to,
    country="India",
    year_from=1951,
    year_to=2001,
    map_col_from="district",
    map_col_to="district",
    exact_match=["state"],
    id_col_from="unit_id",
    id_col_to="unit_id",
    relationship="auto",
    string_exact_match_prune="from",
    evidence=False,
    reason=False,
    model="gemini-3.1-flash-lite-preview",
    gemini_api_key_env="GEMINI_API_KEY",
    replay_enabled=True,
    seed=42,
)

print(crosswalk_df[["from_name", "to_name", "merge", "score"]].head())
print(metadata["artifacts"])
```

6. Review the outputs. By default, AdminLineageAI writes artifacts under `outputs/<country>_<year_from>_<year_to>_<map_col_from>`. The main ones are `evolution_key.csv`, `review_queue.csv`, and `run_metadata.json`.

## Common Options

- `exact_match`: Restricts matching to rows that agree exactly on one or more scope columns such as `country`, `state`, or `district`.
- `string_exact_match_prune`: Controls how aggressively exact string hits are removed from later AI work. Use this to control token spend.
- `relationship`: Declares the kind of relationship you expect, or leave it as `auto`.
- `max_candidates`: Limits how many candidate rows are shown to the model for each source row.
- `evidence`: Adds a short factual summary column.
- `reason`: Adds a longer explanation column.
- `replay_enabled`: Reuses prior completed LLM work when the semantic request matches.
- `seed`: Keeps request identity deterministic for more reproducible reruns.
- `output_dir`: Changes where run artifacts are written.

## Optional CLI Workflow

The CLI is useful when you want a saved YAML config for repeatable runs, but it is optional.

```bash
adminlineage preview --config examples/config/example.yml
adminlineage validate --config examples/config/example.yml
adminlineage run --config examples/config/example.yml
adminlineage export --input outputs/india_1951_2001_subdistrict/evolution_key.csv --format jsonl
```

The package includes these example assets:

- `examples/config/example.yml`
- `examples/loaders/sample_loader.py`
- `examples/adminlineage_gemini_3_1_flash_lite.ipynb`

## Python API

Public objects available from `import adminlineage`:

- `build_evolution_key`
- `preview_plan`
- `validate_inputs`
- `export_crosswalk`
- `get_output_schema_definition`
- `OUTPUT_SCHEMA_VERSION`
- `__version__`

### `build_evolution_key`

Build the evolution key and write run artifacts.

Required arguments:

| Argument | Type | Meaning |
|---|---|---|
| `df_from` | `pd.DataFrame` | Earlier-period table |
| `df_to` | `pd.DataFrame` | Later-period table |
| `country` | `str` | Country label used in prompts and metadata |
| `year_from` | `int \| str` | Earlier-period label |
| `year_to` | `int \| str` | Later-period label |
| `map_col_from` | `str` | Source name column |

Optional arguments:

| Argument | Type | Default | Meaning |
|---|---|---|---|
| `map_col_to` | `str \| None` | `None` | Target name column. Falls back to `map_col_from` when omitted. |
| `exact_match` | `list[str] \| None` | `None` | Columns that must agree before comparison. |
| `id_col_from` | `str \| None` | `None` | Source ID column. |
| `id_col_to` | `str \| None` | `None` | Target ID column. |
| `extra_context_cols` | `list[str] \| None` | `None` | Extra columns added to the model payload. |
| `relationship` | `str` | `auto` | One of `auto`, `father_to_father`, `father_to_child`, `child_to_father`, `child_to_child`. |
| `string_exact_match_prune` | `str` | `none` | `none` keeps exact-string hits in later AI work, `from` removes matched source rows from AI work, `to` removes matched source and target rows from later AI work. |
| `evidence` | `bool` | `False` | Adds a short evidence summary and includes the `evidence` column. |
| `reason` | `bool` | `False` | Adds a longer explanation in the `reason` column. |
| `model` | `str` | `gemini-3.1-flash-lite-preview` | Gemini model name. |
| `gemini_api_key_env` | `str` | `GEMINI_API_KEY` | Environment variable name used for the API key. |
| `batch_size` | `int` | `25` | Maximum number of source rows per Gemini request. When a multi-row request fails, the pipeline retries in smaller batches. |
| `max_candidates` | `int` | `15` | Candidate shortlist size per source row. |
| `output_dir` | `str \| Path` | `outputs` | Base output directory for run artifacts. |
| `seed` | `int` | `42` | Deterministic seed for repeatable request identity. |
| `temperature` | `float` | `0.75` | Gemini temperature. |
| `enable_google_search` | `bool` | `True` | Enables grounded Gemini adjudication. |
| `request_timeout_seconds` | `int \| None` | `90` | Per-request timeout. |
| `env_search_dir` | `str \| Path \| None` | `None` | Starting directory used when searching for `.env`. |
| `replay_enabled` | `bool` | `False` | Reuses prior completed LLM work when the semantic request matches. |
| `replay_store_dir` | `str \| Path \| None` | `None` | Replay store path. Falls back to `.adminlineage_replay` internally when replay is enabled. |

Return value:

- `tuple[pd.DataFrame, dict]`
- first item: the crosswalk DataFrame
- second item: run metadata with counts, warnings, request details, and artifact paths

### `preview_plan`

Preview grouping and candidate-generation behavior without calling Gemini.

```python
adminlineage.preview_plan(
    df_from,
    df_to,
    *,
    country,
    year_from,
    year_to,
    map_col_from,
    map_col_to=None,
    exact_match=None,
    id_col_from=None,
    id_col_to=None,
    extra_context_cols=None,
    string_exact_match_prune="none",
    max_candidates=15,
)
```

Return value: a diagnostics dict describing validity, group sizes, exact-string hits, and candidate budgets.

### `validate_inputs`

Validate the two input tables without running the pipeline.

```python
adminlineage.validate_inputs(
    df_from,
    df_to,
    *,
    country,
    map_col_from,
    map_col_to=None,
    exact_match=None,
    id_col_from=None,
    id_col_to=None,
)
```

Return value: a diagnostics dict that reports whether the inputs are valid and what is missing or duplicated.

### `export_crosswalk`

Convert a materialized crosswalk file into another format.

```python
adminlineage.export_crosswalk(
    input_path="outputs/india_1951_2001_subdistrict/evolution_key.csv",
    output_format="jsonl",
    output_path=None,
)
```

Return value: the written output path.

Supported output formats:

- `csv`
- `parquet`
- `jsonl`

### `get_output_schema_definition`

Return a machine-readable description of the materialized output schema.

```python
schema = adminlineage.get_output_schema_definition(include_evidence=False)
```

Arguments:

| Argument | Type | Default | Meaning |
|---|---|---|---|
| `include_evidence` | `bool` | `False` | Includes the `evidence` column in the returned schema definition. |

Return value: a dict containing the schema version, ordered output columns, required columns, and enum values, including the `merge` indicator enum.

### `OUTPUT_SCHEMA_VERSION`

String constant for the current materialized output schema version.

### `__version__`

String constant for the package version.

## Optional CLI Reference

Commands:

```bash
adminlineage run --config path/to/config.yml
adminlineage preview --config path/to/config.yml
adminlineage validate --config path/to/config.yml
adminlineage export --input path/to/evolution_key.csv --format {csv|parquet|jsonl} [--output path]
```

`preview` and `validate` do not call Gemini. `run` writes the full artifact set. `export` converts an existing materialized crosswalk file. If you are using the Python API directly, you can ignore this section.

## CLI YAML Config Reference

Top-level sections:

- `request`
- `data`
- `llm`
- `pipeline`
- `cache`
- `retry`
- `replay`
- `output`

### `request`

| Key | Default | Meaning |
|---|---|---|
| `country` | required | Country label used in prompts and metadata. |
| `year_from` | required | Earlier-period label. |
| `year_to` | required | Later-period label. |
| `map_col_from` | required | Source name column. |
| `map_col_to` | `null` | Target name column. Falls back to `map_col_from`. |
| `exact_match` | `[]` | Columns that must agree before comparison. |
| `id_col_from` | `null` | Source ID column. |
| `id_col_to` | `null` | Target ID column. |
| `extra_context_cols` | `[]` | Extra columns added to the model payload. |
| `relationship` | `auto` | Relationship mode. |
| `string_exact_match_prune` | `none` | Exact-string pruning mode. |
| `evidence` | `false` | Adds the `evidence` column. |
| `reason` | `false` | Adds the `reason` column. |

### `data`

| Key | Default | Meaning |
|---|---|---|
| `mode` | `files` | One of `files` or `python_hook`. |
| `from_path` | `null` | Required when `mode: files`. |
| `to_path` | `null` | Required when `mode: files`. |
| `callable` | `null` | Required when `mode: python_hook`. Uses `module:function` syntax. |
| `params` | `{}` | Arbitrary config payload passed to the loader hook. |

Loader contract for `python_hook` mode:

```python
def load_data(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    ...
```

The included example hook is `examples/loaders/sample_loader.py`.

For file mode, `data.from_path` and `data.to_path` are resolved relative to the config file location, not your shell location.

### `llm`

| Key | Default | Meaning |
|---|---|---|
| `provider` | `gemini` | Use `gemini` for live runs or `mock` for dry runs and testing. |
| `model` | `gemini-3.1-flash-lite-preview` | Gemini model name. |
| `gemini_api_key_env` | `GEMINI_API_KEY` | Environment variable name for the API key. |
| `temperature` | `0.75` | Gemini temperature. |
| `seed` | `42` | Deterministic seed. |
| `enable_google_search` | `true` | Enables grounded adjudication. |
| `request_timeout_seconds` | `90` | Per-request timeout. |

### `pipeline`

| Key | Default | Meaning |
|---|---|---|
| `batch_size` | `25` | Maximum number of source rows per Gemini request. Failed multi-row requests are retried in smaller batches. |
| `max_candidates` | `15` | Candidate shortlist size per source row. |
| `review_score_threshold` | `0.6` | Rows below this score are flagged for review. |

### `cache`

| Key | Default | Meaning |
|---|---|---|
| `enabled` | `true` | Enables the SQLite LLM cache. |
| `backend` | `sqlite` | Current cache backend. |
| `path` | `llm_cache.sqlite` | Cache database path. |

### `retry`

| Key | Default | Meaning |
|---|---|---|
| `max_attempts` | `6` | Maximum retry attempts for transient LLM failures. |
| `base_delay_seconds` | `1.0` | Initial retry delay. |
| `max_delay_seconds` | `20.0` | Maximum retry delay. |
| `jitter_seconds` | `0.2` | Random jitter added to retry timing. |

### `replay`

| Key | Default | Meaning |
|---|---|---|
| `enabled` | `false` | Enables exact replay for fully completed runs. |
| `store_dir` | `.adminlineage_replay` | Replay bundle directory. |

Relative replay store paths are resolved from the config file location. This section only matters if you are using the CLI workflow.

### `output`

| Key | Default | Meaning |
|---|---|---|
| `write_csv` | `true` | Writes `evolution_key.csv`. |
| `write_parquet` | `true` | Writes `evolution_key.parquet`. |

Minimal config shape:

```yaml
request:
  country: India
  year_from: 1951
  year_to: 2001
  map_col_from: subdistrict
  map_col_to: subdistrict
  exact_match: [state, district]
  id_col_from: unit_id
  id_col_to: unit_id
  relationship: auto
  string_exact_match_prune: none
  evidence: false
  reason: false

data:
  mode: files
  from_path: ../data/from_units.csv
  to_path: ../data/to_units.csv

llm:
  provider: gemini
  model: gemini-3.1-flash-lite-preview
  gemini_api_key_env: GEMINI_API_KEY
  temperature: 0.75
  seed: 42
  enable_google_search: true
  request_timeout_seconds: 90

pipeline:
  batch_size: 25
  max_candidates: 15
  review_score_threshold: 0.6

cache:
  enabled: true
  backend: sqlite
  path: llm_cache.sqlite

retry:
  max_attempts: 6
  base_delay_seconds: 1.0
  max_delay_seconds: 20.0
  jitter_seconds: 0.2

replay:
  enabled: false
  store_dir: .adminlineage_replay

output:
  write_csv: true
  write_parquet: true
```

## Outputs And Utilities

### Main Artifacts

| Artifact | Meaning |
|---|---|
| `evolution_key.csv` | Main crosswalk output. |
| `evolution_key.parquet` | Parquet version of the crosswalk output. |
| `review_queue.csv` | Rows that need manual review. |
| `run_metadata.json` | Run counts, warnings, request details, and artifact paths. |
| `links_raw.jsonl` | Incremental per-row decision log used for resumability and replay publishing. |

### Crosswalk Columns

| Column | Meaning |
|---|---|
| `from_name`, `to_name` | Raw source and target names. |
| `from_canonical_name`, `to_canonical_name` | Normalized names used during matching. |
| `from_id`, `to_id` | User IDs when supplied, otherwise fallback internal IDs. |
| `score` | Confidence in the chosen link, in `[0, 1]`. |
| `link_type` | One of `rename`, `split`, `merge`, `transfer`, `no_match`, `unknown`. |
| `relationship` | One of `father_to_father`, `father_to_child`, `child_to_father`, `child_to_child`, `unknown`. |
| `merge` | `both` for matched rows, `only_in_from` for source-only rows, `only_in_to` for target-only rows appended after the source pass. |
| `evidence` | Short grounded summary. Included only when `evidence=True`. |
| `reason` | Longer explanation. Present as a column, but empty unless `reason=True`. |
| exact-match columns | Copied context columns from the request, such as `state` or `district`. |
| `country`, `year_from`, `year_to` | Request metadata. |
| `run_id` | Deterministic run identifier. |
| `from_key`, `to_key` | Internal stable keys used by the pipeline. |
| `constraints_passed` | Constraint checks recorded for that row. |
| `review_flags`, `review_reason` | QA flags and their comma-joined summary. |

`review_queue.csv` is a filtered subset of the crosswalk for rows that were flagged for manual review. Target-only rows remain in the final evolution key with `merge="only_in_to"`.

## Operational Notes

- `exact_match` scopes the candidate search. If you set `exact_match=["state", "district"]`, a row only compares against rows from the same `(state, district)` group. This is the main hierarchical matching mechanism in the package.
- Candidate generation happens before Gemini. `max_candidates` controls how many shortlist entries the model sees for each source row.
- Exact string handling happens before the model call. `string_exact_match_prune` controls whether already matched rows remain in later AI work.
- Live Gemini work is grounded with Google Search and returns strict JSON. The pipeline then materializes CSV and Parquet outputs itself.
- Replay is opt-in. When `replay_enabled=True`, rerunning the same semantic request reuses the prior completed LLM output instead of calling Gemini again.
- `seed` helps keep request identity deterministic and makes runs easier to reproduce.
- Cache is configured in CLI config. When enabled, the package uses a SQLite cache at `cache.path`.
- Retry behavior is configurable in CLI config. Transient Gemini failures are retried according to the `retry` section before a row is marked unresolved.
- `export_crosswalk` and `adminlineage export` convert an existing materialized crosswalk into `csv`, `parquet`, or `jsonl`.

## A Few Practical Defaults

- `model="gemini-3.1-flash-lite-preview"`
- `temperature=0.75`
- `enable_google_search=True`
- `evidence=False`
- `reason=False`
- `relationship="auto"`
- `string_exact_match_prune="none"`

Those are the current defaults. Change them when you need replay, evidence, stricter scoping, or different review thresholds.

## Citation

If you use AdminLineageAI in published work, please cite:

Siddiqui, T. I., and Vetharenian Hari.
