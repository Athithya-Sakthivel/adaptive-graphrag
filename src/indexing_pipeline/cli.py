"""Thin CLI adapter: argparse, dispatch, exit codes. No pipeline logic."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from . import __version__, db, pipeline
from .io_utils import json_default

EXIT_SUCCESS = 0
EXIT_GATE_FAILED = 1
EXIT_HALTED = 2
EXIT_PRECONDITION = 3
EXIT_CONFIG = 4

logger = logging.getLogger("indexing_pipeline.cli")

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

SUBCOMMANDS = (
    "manifest",
    "chunk",
    "dedup",
    "embed",
    "load",
    "drop-index",
    "validate",
    "index",
    "all",
)


# ── Logging ──────────────────────────────────────────────────────────────


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key in ("stage", "summary", "processed", "skipped", "failed", "elapsed_s"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        return json.dumps(payload, ensure_ascii=False, default=json_default)


def _configure_logging(level: str, json_logs: bool) -> None:
    handler = logging.StreamHandler(sys.stdout)

    if json_logs:
        handler.setFormatter(_JsonFormatter())
    else:

        class _PlainFormatter(logging.Formatter):
            def format(self, record: logging.LogRecord) -> str:
                base = (
                    f"{self.formatTime(record, '%H:%M:%S')} "
                    f"{record.levelname} {record.name}: {record.getMessage()}"
                )
                summary = getattr(record, "summary", None)
                if summary is not None:
                    base += " " + json.dumps(summary, default=json_default)
                return base

        handler.setFormatter(_PlainFormatter())

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        handlers=[handler],
        force=True,
    )


def _log_summary(value: Any) -> None:
    """Log StageReport or list[StageReport] as a structured summary line."""
    if isinstance(value, list):
        payload: Any = [
            asdict(v) if is_dataclass(v) and not isinstance(v, type) else v for v in value
        ]
    elif is_dataclass(value) and not isinstance(value, type):
        payload = asdict(value)
    else:
        payload = value
    logger.info("pipeline stage summary", extra={"stage": "summary", "summary": payload})


# ── Common flags ─────────────────────────────────────────────────────────


def _add_common_flags(p: argparse.ArgumentParser) -> None:
    """Add every flag that any subcommand may need.

    All flags are on every subcommand for uniformity. The pipeline itself
    ignores flags that don't apply to a stage (e.g. --dedup on `validate`),
    so there is no risk of misuse — but there is also no per-command
    restriction to maintain.
    """
    # Paths
    p.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="override data directory (default: <repo>/tmp/data)",
    )
    p.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="override CPU model directory (default: <package>/embedders)",
    )
    p.add_argument(
        "--gpu-model-dir",
        type=Path,
        default=None,
        help="override GPU model directory — base HF checkpoint "
        "(default: <package>/models/dinghy-law-0.6b-v1)",
    )
    p.add_argument(
        "--db-dsn",
        type=str,
        default=None,
        help="override DATABASE_URL",
    )

    # Corpus slice + tuning
    p.add_argument(
        "--corpus-rows",
        type=str,
        default=None,
        help="min | medium | max | positive integer (overrides CORPUS_ROWS)",
    )
    p.add_argument("--workers", type=int, default=None, help="chunk worker processes (1..3)")
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="parquet read batch size (default: 500)",
    )
    p.add_argument(
        "--embed-batch",
        type=int,
        default=None,
        help="per-batch texts passed to encode() (default: 64)",
    )
    p.add_argument(
        "--max-seq-length",
        type=int,
        default=None,
        help="embedder max sequence length (default: 512)",
    )
    p.add_argument(
        "--inference-threads",
        type=int,
        default=None,
        help="OpenVINO CPU inference threads (default: 4)",
    )
    p.add_argument(
        "--pool-min",
        type=int,
        default=None,
        help="DB connection pool minimum (default: 1)",
    )
    p.add_argument(
        "--pool-max",
        type=int,
        default=None,
        help="DB connection pool maximum (default: 4)",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="cpu (default) or cuda / cuda:0 / cuda:0,cuda:1",
    )

    # Execution control
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="validate wiring, write nothing",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="ignore skip logic and _HALTED markers",
    )
    p.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    p.add_argument("--json-logs", action="store_true")

    # Behavior toggles. BooleanOptionalAction gives us --foo / --no-foo pairs.
    # default=None distinguishes "unset" from "explicit False" so Config
    # defaults win when the user does not pass the flag.
    p.add_argument(
        "--dedup",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="cross-case dedup (default: enabled)",
    )
    p.add_argument(
        "--noise-filter",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="drop provable-noise children in chunker (default: enabled)",
    )
    p.add_argument(
        "--strict-tokenizer",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="fail if a chunk worker cannot load a tokenizer "
        "(default: enabled — prevents silent chars/2.25 fallback)",
    )


# ── Argparse ─────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="indexing_pipeline",
        description=f"Legal RAG indexing pipeline v{__version__}",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    for name in SUBCOMMANDS:
        p = sub.add_parser(name, help=f"run the {name} stage")
        _add_common_flags(p)

    return parser


# ── Config construction ──────────────────────────────────────────────────


def _build_config(args: argparse.Namespace) -> pipeline.Config:
    """Build Config from env vars, then apply CLI overrides.

    Only DATABASE_URL and CORPUS_ROWS come from the environment.
    Everything else uses the defaults defined in pipeline_helpers.Config.
    CLI flags override env or defaults when supplied.

    `None` values in the with_overrides call mean "leave unchanged". The
    three BooleanOptionalAction toggles return None when unset, True when
    --foo is passed, False when --no-foo is passed. Config defaults
    survive when the flag was not mentioned.
    """
    config = pipeline.Config.from_env()
    return config.with_overrides(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        gpu_model_dir=args.gpu_model_dir,
        db_dsn=args.db_dsn,
        corpus_rows=args.corpus_rows,
        workers=args.workers,
        batch_size=args.batch_size,
        embed_batch=args.embed_batch,
        max_seq_length=args.max_seq_length,
        inference_threads=args.inference_threads,
        pool_min=args.pool_min,
        pool_max=args.pool_max,
        device=args.device,
        dry_run=args.dry_run or None,
        force=args.force or None,
        dedup_enabled=args.dedup,
        noise_filter_enabled=args.noise_filter,
        strict_worker_tokenizer=args.strict_tokenizer,
    )


# ── Dispatch ─────────────────────────────────────────────────────────────


def _dispatch(args: argparse.Namespace, pl: pipeline.IndexingPipeline) -> int:
    try:
        if args.command == "manifest":
            _log_summary(pl.manifest())
        elif args.command == "chunk":
            _log_summary(pl.chunk())
        elif args.command == "dedup":
            _log_summary(pl.dedup())
        elif args.command == "embed":
            _log_summary(pl.embed())
        elif args.command == "load":
            _log_summary(pl.load())
        elif args.command == "drop-index":
            _log_summary(pl.drop_indexes())
        elif args.command == "validate":
            _log_summary(pl.validate())
        elif args.command == "index":
            _log_summary(pl.index())
        elif args.command == "all":
            _log_summary(pl.run_all())
        else:
            logger.error("unknown command: %s", args.command)
            return EXIT_CONFIG
        return EXIT_SUCCESS

    except pipeline.PipelineError as exc:
        logger.error("%s", exc)
        if exc.reason == "halted":
            return EXIT_HALTED
        if exc.reason == "gate_failed" or exc.reason.startswith("gate_failed:"):
            return EXIT_GATE_FAILED
        if exc.reason.startswith("precondition"):
            return EXIT_PRECONDITION
        return EXIT_CONFIG

    except db.DatabaseError as exc:
        logger.error("%s", exc)
        return EXIT_PRECONDITION

    except FileNotFoundError as exc:
        logger.error("missing input: %s", exc)
        return EXIT_CONFIG

    except ValueError as exc:
        logger.error("config error: %s", exc)
        return EXIT_CONFIG


# ── Entry point ──────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    _configure_logging(args.log_level, args.json_logs)

    try:
        config = _build_config(args)
    except (ValueError, FileNotFoundError, OSError) as exc:
        logger.error("failed to load config: %s", exc)
        return EXIT_CONFIG

    try:
        with pipeline.IndexingPipeline(config) as pl:
            return _dispatch(args, pl)
    except pipeline.PipelineError as exc:
        logger.error("pipeline error: %s", exc)
        return EXIT_HALTED
    except KeyboardInterrupt:
        logger.warning("interrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main())
