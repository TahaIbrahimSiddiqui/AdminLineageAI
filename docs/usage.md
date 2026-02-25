# Usage

## API contract

### `build_evolution_key(...)`

Required arguments:

| Argument | Type | Required | Description |
|---|---|---|---|
| `df_from` | `pd.DataFrame` | yes | Period A units |
| `df_to` | `pd.DataFrame` | yes | Period B units |
| `country` | `str` | yes | Country context in prompt + metadata |
| `year_from` | `int | str` | yes | Period A label |
| `year_to` | `int | str` | yes | Period B label |
| `map_col_from` | `str` | yes | Name column in `df_from` |

Optional arguments:

| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `map_col_to` | `str | None` | no | `None` | Uses `map_col_from` when null |
| `anchor_cols` | `list[str] | None` | no | `None` | Exact-match constraints |
| `id_col_from` | `str | None` | no | `None` | Source ID column |
| `id_col_to` | `str | None` | no | `None` | Target ID column |
| `extra_context_cols` | `list[str] | None` | no | `None` | Additional prompt context columns |
| `aliases` | `pd.DataFrame | None` | no | `None` | Known rename seeds |
| `model` | `str` | no | `gemini-2.5-pro` | Gemini model |
| `gemini_api_key_env` | `str` | no | `GEMINI_API_KEY` | Env var name for API key |
| `batch_size` | `int` | no | `25` | Gemini adjudication batch size |
| `max_candidates` | `int` | no | `15` | Candidate shortlist size |
| `resume_dir` | `str | Path` | no | `outputs` | Base run output folder |
| `run_name` | `str | None` | no | auto | Subfolder under `resume_dir` |
| `seed` | `int` | no | `42` | Deterministic seed |

Return value:

- `tuple[pd.DataFrame, dict]`
- First item is the crosswalk DataFrame.
- Second item is metadata including run info and artifact locations.

## Required and optional columns in user data

### `df_from` columns

| Column | Required | Rule |
|---|---|---|
| `map_col_from` | yes | Must exist |
| each `anchor_cols` member | conditional | Must exist if anchors provided |
| `id_col_from` | no | Used if provided |
| each `extra_context_cols` member | no | Included if provided |

### `df_to` columns

| Column | Required | Rule |
|---|---|---|
| `map_col_to` (or `map_col_from`) | yes | Must exist |
| each `anchor_cols` member | conditional | Must exist if anchors provided |
| `id_col_to` | no | Used if provided |
| each `extra_context_cols` member | no | Included if provided |

### `aliases` columns

| Column | Required | Rule |
|---|---|---|
| `from_alias` | yes | Required when aliases DataFrame provided |
| `to_alias` | yes | Required when aliases DataFrame provided |
| anchor columns | no | Optional scope restriction |

## Meaning of key outputs

### Evolution key (`evolution_key.csv`/`.parquet`)

Each row represents one A->B proposed link.

| Column | Meaning |
|---|---|
| `from_name`, `to_name` | Raw source and target names |
| `from_canonical_name`, `to_canonical_name` | Normalized forms used during matching |
| `from_id`, `to_id` | User IDs or fallback internal IDs |
| `score` | Confidence in `[0,1]` |
| `link_type` | `rename|split|merge|transfer|no_match|unknown` |
| `evidence` | Short summary evidence text |
| anchor columns | Copied context fields from request |
| `country`, `year_from`, `year_to` | Request metadata |
| `run_id` | Deterministic run identifier |
| `from_key`, `to_key` | Internal stable unit keys |
| `constraints_passed` | Constraint check booleans |
| `review_flags`, `review_reason` | Global check outputs for QA |

### Review queue (`review_queue.csv`)

Rows with risk flags such as:

- low score (`score < review_score_threshold`)
- `unknown` or `no_match`
- high fan-in or high fan-out patterns

## Resume behavior

- Run folder path: `resume_dir/run_name`.
- Pipeline appends incremental records to `links_raw.jsonl`.
- On rerun with same folder, completed `from_key` entries are skipped.
- To force a fresh run, use a different `run_name`.

## Gemini API key setup

Set key in shell:

```bash
export GEMINI_API_KEY="your_api_key"
```

Use in config:

```yaml
llm:
  provider: gemini
  model: gemini-2.5-pro
  gemini_api_key_env: GEMINI_API_KEY
```

Custom key name:

```bash
export MY_GEMINI_KEY="your_api_key"
```

```yaml
llm:
  provider: gemini
  gemini_api_key_env: MY_GEMINI_KEY
```

## CLI

```bash
adminlineage preview --config examples/config/example.yml
adminlineage validate --config examples/config/example.yml
adminlineage run --config examples/config/example.yml
adminlineage export --input outputs/example_run/evolution_key.parquet --format csv
```

## Config loader modes

- `data.mode: files` reads `from_path`, `to_path`, optional `aliases_path`.
- `data.mode: python_hook` calls `module:function(config)->(df_from, df_to[, aliases])`.

## Validation workflow

1. Run `validate` before long jobs.
2. Run `preview` to inspect grouping and candidate budget.
3. Inspect `review_queue.csv` after run.
4. Spot-check low-score and `unknown`/`no_match` mappings before downstream use.
