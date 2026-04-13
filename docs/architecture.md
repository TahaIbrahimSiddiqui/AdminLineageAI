# Architecture

`adminlineage` builds administrative evolution keys between two periods with a simple staged pipeline.

## Pipeline Stages

1. **Normalize**
   - Canonicalize names with Unicode normalization, lowercase conversion, punctuation stripping, and whitespace collapse.
   - Build token and character n-gram features for lexical scoring.

2. **Exact-Match Grouping**
   - If `exact_match` is provided, comparisons are restricted to rows with identical tuples in those columns.
   - If `exact_match` is omitted, one global comparison group is used and a warning is emitted.

3. **Candidate Generation**
   - For each source unit, rank target units by token Jaccard plus character n-gram cosine.
   - Keep the top `max_candidates`.

4. **Gemini Adjudication**
   - Send compact JSON payloads with context and shortlists, one source row at a time.
   - Require strict JSON output.
   - Validate responses with pydantic.
   - Retry transient failures and attempt one repair prompt if JSON or schema parsing fails.
   - Optionally request a short `evidence` field when the user enables it.
   - Optionally request a fuller `reason` field when the user enables it.

5. **Global Checks**
   - Compute coverage by exact-match group.
   - Flag low confidence, unknown or no-match rows, and high fan-in or high fan-out patterns.
   - Materialize `review_queue.csv`.

6. **Resumability and Outputs**
   - Incremental batch records append to `links_raw.jsonl`.
   - Re-running the same request in the same working directory reuses the same auto-generated run folder.
   - Final artifacts include:
     - `evolution_key.csv`
     - `evolution_key.parquet` when parquet writing succeeds
     - `review_queue.csv`
     - `run_metadata.json`
     - `run.log`

## LLM Runtime Structure

- `llm_types.py` defines provider interfaces and shared error types.
- `gemini_transport.py` is a direct `google-genai` transport layer for request construction and response extraction.
- `gemini_client.py` holds retry, schema-validation, fallback, and repair policy around that transport.
- `mock_client.py` is used for deterministic tests.

The Gemini client can load a local `.env` file before reading the configured API key environment variable. It does not override environment variables that are already set.

## Pipeline Module Split

- `pipeline.py` orchestrates request identity, replay, artifacts, and metadata.
- `pipeline_stages.py` owns stage execution for first-pass adjudication, second-stage rescue, and row materialization.

## Prompt Contract

The batch prompt requires:

- one decision per `from_key`
- `to_key` limited to the supplied candidates or `null`
- separate `link_type` and `relationship` fields
- `unknown` or `no_match` when evidence is weak
- optional `evidence` only when requested
- optional `reason` only when requested

## Schema Versioning

- Prompt schema version: `2.3.0`
- Output schema version: `2.1.0`

Both versions are persisted in run metadata.
