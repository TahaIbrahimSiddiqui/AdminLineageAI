# Architecture

`adminlineage` builds administrative evolution keys between two periods using a staged pipeline.

## Pipeline Stages

1. **Normalize**
   - Canonicalize names with Unicode normalization, lowercase, punctuation stripping, and whitespace collapse.
   - Build token and character n-gram features for cheap lexical scoring.

2. **Anchor Grouping**
   - If `anchor_cols` are provided, comparisons are restricted to exact anchor tuple matches.
   - If not provided, one global comparison group is used and a warning is emitted.

3. **Candidate Generation**
   - For each `from` unit, rank `to` units by token Jaccard + char n-gram cosine.
   - Optional alias seeds (`from_alias`,`to_alias`, plus optional anchors) boost known rename candidates.
   - Keep top `max_candidates`.

4. **Gemini Adjudication**
   - Batch `from` units and send compact prompt payloads with context and shortlists.
   - Require strict JSON schema output.
   - Validate responses with pydantic.
   - Retry transient failures and run one repair prompt if JSON/schema parse fails.

5. **Global Checks**
   - Compute coverage by anchor group.
   - Flag low confidence, unknown/no-match, and high fan-in/fan-out patterns.
   - Materialize `review_queue.csv`.

6. **Resumability + Outputs**
   - Incremental batch records append to `links_raw.jsonl`.
   - Reruns skip already completed `from_key` units.
   - Final artifacts:
     - `evolution_key.parquet` (when parquet engine is available)
     - `evolution_key.csv`
     - `review_queue.csv`
     - `run_metadata.json`

## LLM Abstraction

- `BaseLLMClient.generate_json(...)`
- `GeminiClient` for production runs
- `MockClient` for deterministic tests

LLM cache defaults to SQLite (`llm_cache.sqlite`) keyed by `(model, prompt_hash, schema_version)`.

## Schema Versioning

- Prompt schema version: `1.0.0`
- Output schema version: `1.0.0`

Both versions are persisted in run metadata.
