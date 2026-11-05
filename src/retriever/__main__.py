#!/usr/bin/env python3
"""Command-line entry point for Adaptive GraphRAG retrieval.

The retriever is CPU-only by design. Embedding uses
``retriever.embedder.QueryEmbedder`` — a batch-1, sequence-128
OpenVINO model compiled with the LATENCY hint. LOOKUP queries never
load the embedder; HOLDING, VALIDITY, and APPLICATION do (dense arm).

Pre-hardening CLI imported ``indexing_pipeline.embedder.Embedder``
(batch-8, seq-464, THROUGHPUT), paying 8× batch waste and ~3.6×
sequence waste on every query — 24 s/query on the query path versus
~1 s with ``QueryEmbedder``. The GPU/CUDA path is retired: the
retrieval load profile is single-query CPU, not batch GPU.

``--config-override KEY=VALUE`` (repeatable) lets eval scripts sweep
``RetrieverConfig`` fields without editing source::

    python -m retriever "query" --config-override dense_k=10
    python -m retriever "query" --config-override case_diversity_enabled=false
    python -m retriever "query" --config-override graph_rank_by=case_id

Field names must match a ``RetrieverConfig`` dataclass field.
Values are coerced from the raw string based on the field's default
type: ``bool`` accepts true/false/1/0/yes/no/on/off, ``int`` and
``float`` use standard parsing, everything else is passed through
as a string.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections.abc import Sequence
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any

import psycopg

from .config import RetrieverConfig, default_config
from .filters import Filters
from .router import route

logger = logging.getLogger("retriever.cli")


_SRC_DIR = Path(__file__).resolve().parents[1]

_DEFAULT_MODEL_DIR = (
    _SRC_DIR / "indexing_pipeline" / "embedders" / "dinghy-law-0.6b-v1-openvino-fp16"
)

# QueryEmbedder defaults; kept here so help text and LazyEmbedder
# agree without importing retriever.embedder at CLI load time.
_DEFAULT_SEQUENCE_LENGTH = 128
_DEFAULT_THREADS = 4


# ── Argument parsing helpers ───────────────────────────────────────────


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid date {value!r}; expected YYYY-MM-DD") from exc


def _parse_config_override(raw: str) -> tuple[str, object]:
    """Parse ``KEY=VALUE`` into a typed (key, value) tuple.

    Uses the field's default-value type from ``RetrieverConfig()`` to
    decide coercion. Raises ``ValueError`` on malformed input, unknown
    keys, or type mismatches; the caller (``main``) turns that into a
    clean "invalid request" exit code.
    """
    if "=" not in raw:
        raise ValueError(f"--config-override requires KEY=VALUE format, got {raw!r}")

    key, _, value_str = raw.partition("=")
    key = key.strip()
    value_str = value_str.strip()

    if not key:
        raise ValueError(f"--config-override has empty key: {raw!r}")

    if key not in RetrieverConfig.__dataclass_fields__:
        raise ValueError(f"unknown config field: {key!r}; see RetrieverConfig for valid names")

    reference = RetrieverConfig()  # class defaults — type inspection only
    current = getattr(reference, key)

    if isinstance(current, bool):
        low = value_str.lower()
        if low in ("true", "1", "yes", "on"):
            return key, True
        if low in ("false", "0", "no", "off"):
            return key, False
        raise ValueError(
            f"invalid bool value for {key!r}: {value_str!r} (accepts true/false/1/0/yes/no/on/off)"
        )

    if isinstance(current, int):
        try:
            return key, int(value_str)
        except ValueError as exc:
            raise ValueError(f"invalid int for {key!r}: {value_str!r}") from exc

    if isinstance(current, float):
        try:
            return key, float(value_str)
        except ValueError as exc:
            raise ValueError(f"invalid float for {key!r}: {value_str!r}") from exc

    return key, value_str


def _apply_overrides(config: RetrieverConfig, raw_overrides: list[str] | None) -> RetrieverConfig:
    """Apply all ``--config-override`` args to ``config`` in one shot.

    Combines into a single ``with_overrides(**kwargs)`` call so that
    validation runs once, atomically, and errors name every bad key
    rather than stopping at the first.
    """
    if not raw_overrides:
        return config
    parsed = [_parse_config_override(raw) for raw in raw_overrides]
    return config.with_overrides(**dict(parsed))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="retriever",
        description="Adaptive GraphRAG legal research retrieval.",
    )

    parser.add_argument(
        "query",
        help="Natural-language US case-law query",
    )

    parser.add_argument(
        "--db-dsn",
        default=os.environ.get("DATABASE_URL"),
        help="PostgreSQL DSN; defaults to DATABASE_URL",
    )

    parser.add_argument(
        "--model-dir",
        type=Path,
        default=_DEFAULT_MODEL_DIR,
        help="Directory containing the OpenVINO QueryEmbedder export",
    )

    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=None,
        help=(
            "Static sequence length compiled into the QueryEmbedder "
            f"(default {_DEFAULT_SEQUENCE_LENGTH}); queries longer "
            "than this raise ValueError rather than truncating"
        ),
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=_DEFAULT_THREADS,
        help=(
            "inference_num_threads for OpenVINO CPU inference "
            f"(default {_DEFAULT_THREADS}; use physical cores, "
            "not logical)"
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--court",
        default=None,
    )

    parser.add_argument(
        "--jurisdiction",
        default=None,
    )

    parser.add_argument(
        "--published-only",
        action="store_true",
    )

    parser.add_argument(
        "--date-from",
        type=_parse_date,
        default=None,
    )

    parser.add_argument(
        "--date-to",
        type=_parse_date,
        default=None,
    )

    parser.add_argument(
        "--case-id",
        type=int,
        action="append",
        default=None,
        help="Restrict to a case ID; repeat to supply multiple IDs",
    )

    parser.add_argument(
        "--config-override",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help=(
            "Override a RetrieverConfig field (repeatable). Example: --config-override dense_k=10"
        ),
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON to stdout",
    )

    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )

    return parser


# ── Lazy embedder wrapper ──────────────────────────────────────────────


class LazyEmbedder:
    """Load ``QueryEmbedder`` only when ``embed_query()`` is called.

    LOOKUP queries never invoke the embedder, so they pay zero model
    load cost — this is the primary reason the retriever can answer
    "which cases cite X" in well under a second.

    Attribute access (``.tokenizer``, ``.max_seq_length``) delegates
    to the underlying instance via ``__getattr__``, so the router's
    no-truncation guard sees the real tokenizer after a load is
    triggered.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self._instance: Any = None

    def _get(self) -> Any:
        if self._instance is not None:
            return self._instance

        from .embedder import QueryEmbedder

        sequence_length = (
            _DEFAULT_SEQUENCE_LENGTH
            if self.args.max_seq_length is None
            else self.args.max_seq_length
        )

        if sequence_length < 16:
            raise ValueError(f"--max-seq-length must be >= 16, got {sequence_length}")

        if self.args.threads < 1:
            raise ValueError("--threads must be positive")

        logger.info(
            "loading QueryEmbedder: model=%s seq=%d threads=%d",
            self.args.model_dir,
            sequence_length,
            self.args.threads,
        )

        self._instance = QueryEmbedder(
            model_dir=self.args.model_dir.expanduser().resolve(),
            sequence_length=sequence_length,
            threads=self.args.threads,
        )
        return self._instance

    def embed_query(self, text: str) -> Any:
        return self._get().embed_query(text)

    def __getattr__(self, name: str) -> Any:
        # Delayed delegation — only triggers a load when the caller
        # actually needs an attribute that isn't on LazyEmbedder itself.
        return getattr(self._get(), name)

    def close(self) -> None:
        if self._instance is None:
            return
        instance = self._instance
        self._instance = None
        try:
            instance.close()
        except Exception:
            logger.exception("failed to close QueryEmbedder")


# ── Filters ────────────────────────────────────────────────────────────


def _build_filters(args: argparse.Namespace) -> Filters:
    return Filters(
        court=args.court,
        jurisdiction=args.jurisdiction,
        published_only=args.published_only,
        date_from=args.date_from,
        date_to=args.date_to,
        case_ids=args.case_id,
    )


# ── Output formatting ──────────────────────────────────────────────────


def _print_text(result: Any) -> None:
    print()
    print(f"Query: {result.query}")
    print(f"Class: {result.query_class.value}")
    print(f"Time:  {result.latency_ms:.2f} ms")

    if result.arm_diagnostics:
        print()
        print("Arms:")
        for d in result.arm_diagnostics:
            suffix = ""
            if d.error:
                suffix = f"  ERROR: {d.error}"
            elif not d.attempted:
                suffix = "  (not attempted)"
            print(f"  {d.name:<22} hits={d.hits:<5} {d.latency_ms:>8.1f} ms{suffix}")
    else:
        print()
        print("Arms: none")

    print()

    if not result.hits:
        print("No results.")
        return

    for hit in result.hits:
        flag = "P" if hit.is_parent else "C"
        preview = str(hit.meta.get("preview") or "").replace("\n", " ").strip()

        print(
            f"[{hit.rank:2d}] [{flag}] {hit.chunk_id} case_id={hit.case_id} score={hit.score:.6f}"
        )
        print(f"     Citation: {hit.meta.get('citation') or 'N/A'}")
        print(f"     Court:    {hit.meta.get('court') or 'N/A'}")

        if hit.meta.get("hop") is not None:
            print(f"     Graph hop: {hit.meta['hop']}")

        print(f"     {preview[:160]}")
        print()


def _print_json(result: Any) -> None:
    payload = {
        "query": result.query,
        "query_class": result.query_class.value,
        "arms_used": list(result.arms_used),
        "latency_ms": round(result.latency_ms, 3),
        "arm_diagnostics": [asdict(d) for d in result.arm_diagnostics],
        "hits": [asdict(hit) for hit in result.hits],
    }

    print(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )


# ── Entry point ────────────────────────────────────────────────────────


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.db_dsn or not args.db_dsn.strip():
        print(
            "error: DATABASE_URL is missing; provide --db-dsn or export DATABASE_URL",
            file=sys.stderr,
        )
        return 4

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )

    embedder = LazyEmbedder(args)

    try:
        filters = _build_filters(args)

        if args.config_override:
            config = _apply_overrides(default_config(), args.config_override)
        else:
            config = None  # route() uses default_config() internally

        # Instantiate no embedding model here. LazyEmbedder creates the
        # model only when a route actually needs dense retrieval.
        with psycopg.connect(args.db_dsn) as conn:
            result = route(
                conn,
                embedder,
                args.query,
                filters=filters,
                top_k=args.top_k,
                config=config,
            )

        if args.json:
            _print_json(result)
        else:
            _print_text(result)

        return 0

    except (ValueError, TypeError) as exc:
        logger.error("invalid request/configuration: %s", exc)
        print(f"error: {exc}", file=sys.stderr)
        return 2

    except (psycopg.Error, OSError, RuntimeError) as exc:
        logger.exception("retrieval failed")
        print(
            f"error: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1

    except Exception as exc:
        logger.exception("unexpected retrieval failure")
        print(
            f"error: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1

    finally:
        embedder.close()


if __name__ == "__main__":
    sys.exit(main())
