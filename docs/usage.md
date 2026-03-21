# Usage

## API Contract

### `build_evolution_key(...)`

Required arguments:

| Argument | Type | Required | Description |
|---|---|---|---|
| `df_from` | `pd.DataFrame` | yes | Earlier-period units |
| `df_to` | `pd.DataFrame` | yes | Later-period units |
| `country` | `str` | yes | Country context in prompt and metadata |
| `year_from` | `int | str` | yes | Earlier-period label |
| `year_to` | `int | str` | yes | Later-period label |
| `map_col_from` | `str` | yes | Source name column |

Common optional arguments:

| Argument | Type | Default | Description |
|---|---|---|---|
| `map_col_to` | `str | None` | `None` | Uses `map_col_from` when not provided |
| `exact_match` | `list[str] | None` | `None` | Columns that must match exactly before comparison |
| `id_col_from` | `str | None` | `None` | Source ID column |
| `id_col_to` | `str | None` | `None` | Target ID column |
| `extra_context_cols` | `list[str] | None` | `None` | Extra columns included in the prompt payload |
| `relationship` | `str` | `auto` | Relationship mode |
| `reason` | `bool` | `False` | Ask the model for a fuller explanation |
| `model` | `str` | `gemini-2.5-pro` | Gemini model |
| `gemini_api_key_env` | `str` | `GEMINI_API_KEY` | Environment variable name for the API key |
| `batch_size` | `int` | `25` | LLM adjudication batch size |
| `max_candidates` | `int` | `15` | Candidate shortlist size |
| `seed` | `int` | `42` | Deterministic seed |

Return value:

- `tuple[pd.DataFrame, dict]`
- First item is the crosswalk DataFrame.
- Second item is metadata with counts, warnings, request details, and artifact paths.

## Relationship Values

Allowed request values:

- `auto`
- `father_to_father`
- `father_to_child`
- `child_to_father`
- `child_to_child`

Output rows may contain:

- `father_to_father`
- `father_to_child`
- `child_to_father`
- `child_to_child`
- `unknown`

## Required and Optional Columns in User Data

### `df_from`

| Column | Required | Rule |
|---|---|---|
| `map_col_from` | yes | Must exist |
| each `exact_match` member | conditional | Must exist if `exact_match` is provided |
| `id_col_from` | no | Used if provided |
| each `extra_context_cols` member | no | Included if provided |

### `df_to`

| Column | Required | Rule |
|---|---|---|
| `map_col_to` or `map_col_from` | yes | Must exist |
| each `exact_match` member | conditional | Must exist if `exact_match` is provided |
| `id_col_to` | no | Used if provided |
| each `extra_context_cols` member | no | Included if provided |

## Meaning of Key Outputs

### Evolution key (`evolution_key.csv` / `.parquet`)

Each row represents one proposed link from period A to period B.

| Column | Meaning |
|---|---|
| `from_name`, `to_name` | Raw source and target names |
| `from_canonical_name`, `to_canonical_name` | Normalized forms used during matching |
| `from_id`, `to_id` | User IDs or fallback internal IDs |
| `score` | Confidence in `[0,1]` |
| `link_type` | `rename|split|merge|transfer|no_match|unknown` |
| `relationship` | Hierarchical relationship for the link |
| `evidence` | Short factual summary |
| `reason` | Optional fuller explanation |
| exact-match columns | Copied context fields from the request |
| `country`, `year_from`, `year_to` | Request metadata |
| `run_id` | Deterministic run identifier |
| `from_key`, `to_key` | Internal stable unit keys |
| `constraints_passed` | Constraint check booleans |
| `review_flags`, `review_reason` | Global QA outputs |

### Review queue (`review_queue.csv`)

Rows with flags such as:

- low score (`score < review_score_threshold`)
- `unknown` or `no_match`
- high fan-in or high fan-out patterns

## Run Location and Resumability

- Runs are written to `outputs/<auto_run_name>` under the current working directory.
- `links_raw.jsonl` is written incrementally and used for resumability.
- Re-running the same request in the same working directory reuses the same run folder.

## Gemini API Key Setup

Preferred local setup:

```bash
cp .env.example .env
```

Then put your key in `.env`:

```bash
GEMINI_API_KEY=your_api_key
```

You can also use a custom variable name:

```yaml
llm:
  gemini_api_key_env: MY_GEMINI_KEY
```

## CLI

```bash
adminlineage preview --config examples/config/example.yml
adminlineage validate --config examples/config/example.yml
adminlineage run --config examples/config/example.yml
adminlineage export --input outputs/india_1951_2001_subdistrict/evolution_key.csv --format jsonl
```

## Config Loader Modes

- `data.mode: files` reads `from_path` and `to_path`
- `data.mode: python_hook` calls `module:function(config)->(df_from, df_to)`

For file mode, relative paths are resolved relative to the config file.

## Validation Workflow

1. Run `validate` before long jobs.
2. Run `preview` to inspect grouping and candidate budget.
3. Inspect `review_queue.csv` after the run.
4. Spot-check low-score and `unknown` / `no_match` mappings before using the output downstream.
