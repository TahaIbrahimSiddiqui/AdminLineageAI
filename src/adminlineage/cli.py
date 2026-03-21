"""Command-line interface for adminlineage."""

from __future__ import annotations

import argparse
import json
import sys

from .config import load_config
from .export import export_crosswalk_file
from .io import load_frames
from .llm import MockClient
from .pipeline import preview_pipeline_plan, run_pipeline
from .validation import validate_inputs_data


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="adminlineage")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run evolution-key pipeline")
    run_parser.add_argument("--config", required=True, help="Path to YAML config")

    preview_parser = subparsers.add_parser("preview", help="Preview candidate plan")
    preview_parser.add_argument("--config", required=True, help="Path to YAML config")

    validate_parser = subparsers.add_parser("validate", help="Validate config and data")
    validate_parser.add_argument("--config", required=True, help="Path to YAML config")

    export_parser = subparsers.add_parser("export", help="Convert crosswalk output format")
    export_parser.add_argument("--input", required=True, help="Input CSV/Parquet crosswalk")
    export_parser.add_argument("--format", required=True, choices=["csv", "parquet", "jsonl"])
    export_parser.add_argument("--output", required=False)

    return parser


def _run_from_config(config_path: str) -> int:
    cfg = load_config(config_path)
    loaded = load_frames(cfg)

    llm_client = None
    if cfg.llm.provider == "mock":
        llm_client = MockClient()

    crosswalk, metadata = run_pipeline(
        loaded.df_from,
        loaded.df_to,
        country=cfg.request.country,
        year_from=cfg.request.year_from,
        year_to=cfg.request.year_to,
        map_col_from=cfg.request.map_col_from,
        map_col_to=cfg.request.map_col_to,
        exact_match=cfg.request.exact_match,
        id_col_from=cfg.request.id_col_from,
        id_col_to=cfg.request.id_col_to,
        extra_context_cols=cfg.request.extra_context_cols,
        relationship=cfg.request.relationship,
        reason=cfg.request.reason,
        model=cfg.llm.model,
        gemini_api_key_env=cfg.llm.gemini_api_key_env,
        batch_size=cfg.pipeline.batch_size,
        max_candidates=cfg.pipeline.max_candidates,
        seed=cfg.llm.seed,
        llm_client=llm_client,
        temperature=cfg.llm.temperature,
        cache_enabled=cfg.cache.enabled,
        cache_path=cfg.cache.path,
        retry_max_attempts=cfg.retry.max_attempts,
        retry_base_delay=cfg.retry.base_delay_seconds,
        retry_max_delay=cfg.retry.max_delay_seconds,
        retry_jitter=cfg.retry.jitter_seconds,
        review_score_threshold=cfg.pipeline.review_score_threshold,
        output_write_csv=cfg.output.write_csv,
        output_write_parquet=cfg.output.write_parquet,
        loader_metadata=loaded.loader_metadata,
        env_search_dir=cfg.source_dir,
    )
    _ = crosswalk

    print(
        json.dumps(
            {
                "status": "ok",
                "run_id": metadata["run_id"],
                "artifacts": metadata["artifacts"],
            },
            indent=2,
        )
    )
    return 0


def _preview_from_config(config_path: str) -> int:
    cfg = load_config(config_path)
    loaded = load_frames(cfg)

    preview = preview_pipeline_plan(
        loaded.df_from,
        loaded.df_to,
        country=cfg.request.country,
        year_from=cfg.request.year_from,
        year_to=cfg.request.year_to,
        map_col_from=cfg.request.map_col_from,
        map_col_to=cfg.request.map_col_to,
        exact_match=cfg.request.exact_match,
        id_col_from=cfg.request.id_col_from,
        id_col_to=cfg.request.id_col_to,
        extra_context_cols=cfg.request.extra_context_cols,
        max_candidates=cfg.pipeline.max_candidates,
    )
    print(json.dumps(preview, indent=2))
    return 0 if preview["valid"] else 2


def _validate_from_config(config_path: str) -> int:
    cfg = load_config(config_path)
    loaded = load_frames(cfg)

    diagnostics = validate_inputs_data(
        loaded.df_from,
        loaded.df_to,
        country=cfg.request.country,
        map_col_from=cfg.request.map_col_from,
        map_col_to=cfg.request.map_col_to,
        exact_match=cfg.request.exact_match,
        id_col_from=cfg.request.id_col_from,
        id_col_to=cfg.request.id_col_to,
    )
    print(json.dumps(diagnostics, indent=2))
    return 0 if diagnostics["valid"] else 2


def _export_crosswalk(args: argparse.Namespace) -> int:
    output_path = export_crosswalk_file(args.input, args.format, args.output)
    print(str(output_path))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        return _run_from_config(args.config)
    if args.command == "preview":
        return _preview_from_config(args.config)
    if args.command == "validate":
        return _validate_from_config(args.config)
    if args.command == "export":
        return _export_crosswalk(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
