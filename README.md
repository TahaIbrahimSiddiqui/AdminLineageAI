# adminlineage

`adminlineage` helps you build an administrative evolution key between two time periods.

In plain terms, you give it one table from an older period, one table from a newer period, and the name columns you want to match. It scores candidate matches, asks Gemini to choose the most plausible links, and writes a crosswalk you can review.

The supported workflow in this repo is one path only:

- Gemini `gemini-3.1-flash-lite-preview`
- Google Search grounding enabled
- strict JSON output schema
- sequential row-by-row adjudication for reliability

The package is meant to be easy to pick up:

- one install command
- one `.env` file for the API key
- one small pandas example
- one small CLI config
- sensible defaults so you can get a first run working quickly

## What It Produces

The main output is an `evolution_key.csv` file. Each row is one proposed link from a source unit to a target unit, with:

- `link_type` such as `rename`, `split`, `merge`, `transfer`, `no_match`, or `unknown`
- `relationship` such as `father_to_father` or `father_to_child`
- an optional short `evidence` summary when `evidence=True`
- an optional `reason` field if you explicitly turn it on
- `review_flags` and `review_reason` for manual QA

It also writes a `review_queue.csv`, a `run_metadata.json`, and an internal `links_raw.jsonl` file used for resumability.

## Install

```bash
python -m pip install -e .[dev,io]
```

## 2-Minute Setup

Create a `.env` file in the repo root:

```bash
GEMINI_API_KEY=your_api_key_here
```

You can use the included template too:

```bash
cp .env.example .env
```

The package will load `.env` automatically without overwriting an API key you already exported in your shell.

## Quickstart With Pandas

```python
import pandas as pd
import adminlineage

from_df = pd.read_csv("examples/data/from_units.csv")
to_df = pd.read_csv("examples/data/to_units.csv")

crosswalk_df, metadata = adminlineage.build_evolution_key(
    from_df,
    to_df,
    country="India",
    year_from=1951,
    year_to=2001,
    map_col_from="subdistrict",
    map_col_to="subdistrict",
    exact_match=["state", "district"],
    id_col_from="unit_id",
    id_col_to="unit_id",
    relationship="auto",
    evidence=False,
    reason=False,
    model="gemini-3.1-flash-lite-preview",
    gemini_api_key_env="GEMINI_API_KEY",
)

print(crosswalk_df.head())
print(metadata["artifacts"])
```

By default, outputs are written under `outputs/<auto_run_name>` in your current working directory.

## Quickstart With The CLI

The example config is ready to use:

```bash
adminlineage preview --config examples/config/example.yml
adminlineage validate --config examples/config/example.yml
adminlineage run --config examples/config/example.yml
adminlineage export --input outputs/india_1951_2001_subdistrict/evolution_key.csv --format jsonl
```

`from_path` and `to_path` in the config are resolved relative to the config file, not your shell location. That makes it much easier to run the CLI from anywhere once the config is in place.

## Minimal CLI Config

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

pipeline:
  batch_size: 1
  max_candidates: 15
```

## Core Inputs

Required API arguments:

| Argument | Meaning |
|---|---|
| `df_from` | DataFrame for the earlier period |
| `df_to` | DataFrame for the later period |
| `country` | Country label used in prompts and metadata |
| `year_from` | Label for the earlier period |
| `year_to` | Label for the later period |
| `map_col_from` | Name column in `df_from` to map |

Common optional arguments:

| Argument | Meaning |
|---|---|
| `map_col_to` | Name column in `df_to`; defaults to `map_col_from` |
| `exact_match` | Columns that must match exactly before candidates are compared |
| `id_col_from` | Stable ID column in `df_from` |
| `id_col_to` | Stable ID column in `df_to` |
| `extra_context_cols` | Extra columns included in the model payload |
| `relationship` | `auto` by default, or one of the explicit relationship modes |
| `evidence` | `False` by default; when `True`, asks the model for a short factual summary |
| `reason` | `False` by default; when `True`, asks the model for a fuller explanation |
| `model` | Gemini model name |
| `gemini_api_key_env` | Environment variable name containing the API key |
| `batch_size` | Compatibility setting; live Gemini runs execute sequentially |
| `max_candidates` | Candidate shortlist size |
| `seed` | Deterministic seed for repeatable runs |

## Input Expectations

### `df_from`

| Column type | Required | Notes |
|---|---|---|
| `map_col_from` | yes | Source unit name |
| each `exact_match` column | conditional | Required only if `exact_match` is provided |
| `id_col_from` | no | Optional stable identifier |
| each `extra_context_cols` member | no | Optional extra context |

### `df_to`

| Column type | Required | Notes |
|---|---|---|
| `map_col_to` or `map_col_from` | yes | Target unit name |
| each `exact_match` column | conditional | Required only if `exact_match` is provided |
| `id_col_to` | no | Optional stable identifier |
| each `extra_context_cols` member | no | Optional extra context |

## The Two Settings Most People Care About

### `exact_match`

Use `exact_match` when some parent-level columns should agree before a comparison is even allowed.

Example:

```python
exact_match=["state", "district"]
```

That means a source row in `(S1, D1)` will only be compared against target rows in `(S1, D1)`.

If you leave `exact_match` empty, the tool compares everything globally. That can be useful for rough exploration, but false positives are more likely.

### `relationship`

`relationship` tells the pipeline how to think about the hierarchical relation between matched units.

Allowed values:

- `auto`
- `father_to_father`
- `father_to_child`
- `child_to_father`
- `child_to_child`

When you use `auto`, the model infers the relationship and writes it to the output. If you choose one explicit value, matched links are constrained to that relationship and the same value is written into the result rows.

## Optional `evidence`

`evidence` is off by default:

```python
evidence=False
```

That keeps the structured output smaller and avoids paying for short factual summaries unless you want them.

If you set:

```python
evidence=True
```

the package asks the model for a short factual summary and includes an `evidence` column in the crosswalk output.

## Optional `reason`

`reason` is off by default:

```python
reason=False
```

That keeps the prompt leaner and the token bill lower.

If you set:

```python
reason=True
```

the package asks the model for a fuller explanation and writes it to the `reason` column.

Use it when you want more traceability. Skip it when you want faster, cheaper runs. The cost can go up noticeably on larger jobs.

## Output Files

For each run directory, you should expect:

- `evolution_key.csv`
- `evolution_key.parquet` if a parquet engine is available and writing is enabled
- `review_queue.csv`
- `run_metadata.json`
- `links_raw.jsonl`
- `run.log`

## Output Columns

Key columns in the crosswalk:

| Column | Meaning |
|---|---|
| `from_name`, `to_name` | Raw source and target names |
| `from_canonical_name`, `to_canonical_name` | Normalized names used in matching |
| `from_id`, `to_id` | User IDs or internal fallback IDs |
| `score` | Confidence score in `[0,1]` |
| `link_type` | Match type |
| `relationship` | Hierarchical relationship for the link |
| `evidence` | Short factual summary, included only when `evidence=True` |
| `reason` | Optional fuller explanation |
| `constraints_passed` | Checks like `candidate_membership` and `exact_match` |
| each `exact_match` column | Copied into the output for context |
| `review_flags`, `review_reason` | Manual review hints |

## How The Matching Works

The pipeline is straightforward:

1. Normalize names.
2. Group rows by `exact_match` when provided.
3. Build a lexical shortlist for each source row.
4. Ask Gemini 3.1 Flash-Lite to choose links from that shortlist only, one source row at a time, with Google Search grounding limited to shortlist verification.
5. Write outputs and flag questionable rows for review.

This is a model-assisted workflow, not an automatic truth machine. You still need to read `review_queue.csv` and spot-check important cases.

## Troubleshooting

### “Missing Gemini API key”

- Make sure `.env` exists in the repo root, or
- export the variable named in `gemini_api_key_env`

### Preview or validate fails

- Check that every `exact_match` column exists in both dataframes
- Check that `map_col_from` and `map_col_to` point to real columns
- Check that the config file paths are correct relative to the config file

### The run looks too broad

- Add or tighten `exact_match`
- Lower `max_candidates`
- Use a more specific `map_col_to` if the target name column differs

### The results feel too thin

- Turn on `evidence=True`
- Turn on `reason=True`
- Add a few `extra_context_cols`
- Review borderline rows in `review_queue.csv`

## Documentation

- [`docs/architecture.md`](docs/architecture.md)
- [`docs/usage.md`](docs/usage.md)
- [`docs/output_schema.json`](docs/output_schema.json)

## Development

```bash
ruff check .
python -m pytest
python -m build
```
