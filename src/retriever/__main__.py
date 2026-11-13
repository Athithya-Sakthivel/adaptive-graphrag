"""CLI entry point for Adaptive GraphRAG retrieval.

CPU-only by design. QueryEmbedder loads lazily on first dense query,
so LOOKUP queries (pure graph/exact) pay zero model load cost.
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
_DEFAULT_MODEL_DIR = _SRC_DIR / "indexing_pipeline" / "embedders"
_DEFAULT_SEQUENCE_LENGTH = 128
_DEFAULT_THREADS = 4


class LazyEmbedder:
    """Defer QueryEmbedder load until first embed_query() call."""

    def __init__(self, model_dir: Path, sequence_length: int, threads: int) -> None:
        self.model_dir = model_dir
        self.sequence_length = sequence_length
        self.threads = threads
        self._instance: Any = None

    def _get(self) -> Any:
        if self._instance is None:
            from .embedder import QueryEmbedder

            logger.info(
                "loading QueryEmbedder: model=%s seq=%d threads=%d",
                self.model_dir.name,
                self.sequence_length,
                self.threads,
            )
            self._instance = QueryEmbedder(
                model_dir=self.model_dir,
                sequence_length=self.sequence_length,
                threads=self.threads,
            )
        return self._instance

    def embed_query(self, text: str) -> Any:
        return self._get().embed_query(text)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._get(), name)

    def close(self) -> None:
        if self._instance is not None:
            try:
                self._instance.close()
            except Exception:
                logger.exception("failed to close QueryEmbedder")
            self._instance = None


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid date {value!r}; expected YYYY-MM-DD") from exc


def _parse_config_override(raw: str) -> tuple[str, Any]:
    if "=" not in raw:
        raise ValueError(f"--config-override requires KEY=VALUE format, got {raw!r}")

    key, _, value_str = raw.partition("=")
    key = key.strip()
    value_str = value_str.strip()

    if not key:
        raise ValueError(f"--config-override has empty key: {raw!r}")

    reference = RetrieverConfig()
    if key not in reference.__dataclass_fields__:
        raise ValueError(f"unknown config field: {key!r}")

    current = getattr(reference, key)

    if isinstance(current, bool):
        low = value_str.lower()
        if low in ("true", "1", "yes", "on"):
            return key, True
        if low in ("false", "0", "no", "off"):
            return key, False
        raise ValueError(f"invalid bool for {key!r}: {value_str!r}")

    if isinstance(current, int):
        return key, int(value_str)
    if isinstance(current, float):
        return key, float(value_str)
    return key, value_str


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="retriever",
        description="Adaptive GraphRAG legal research retrieval.",
    )
    parser.add_argument("query", help="Natural-language US case-law query")
    parser.add_argument("--db-dsn", default=os.environ.get("DATABASE_URL"))
    parser.add_argument("--model-dir", type=Path, default=_DEFAULT_MODEL_DIR)
    parser.add_argument("--max-seq-length", type=int, default=_DEFAULT_SEQUENCE_LENGTH)
    parser.add_argument("--threads", type=int, default=_DEFAULT_THREADS)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--court", default=None)
    parser.add_argument("--jurisdiction", default=None)
    parser.add_argument("--published-only", action="store_true")
    parser.add_argument("--date-from", type=_parse_date, default=None)
    parser.add_argument("--date-to", type=_parse_date, default=None)
    parser.add_argument("--case-id", type=int, action="append", default=None)
    parser.add_argument(
        "--config-override",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="Override RetrieverConfig field (repeatable)",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON to stdout")
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser


def _print_text(result: Any) -> None:
    print(f"\nQuery: {result.query}")
    print(f"Class: {result.query_class.value}")
    print(f"Time:  {result.latency_ms:.2f} ms\n")

    if result.arm_diagnostics:
        print("Arms:")
        for d in result.arm_diagnostics:
            suffix = ""
            if d.error:
                suffix = f"  ERROR: {d.error}"
            elif not d.attempted:
                suffix = "  (not attempted)"
            print(f"  {d.name:<22} hits={d.hits:<5} {d.latency_ms:>8.1f} ms{suffix}")
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
        print(f"     {preview[:160]}\n")


def _print_json(result: Any) -> None:
    payload = {
        "query": result.query,
        "query_class": result.query_class.value,
        "arms_used": list(result.arms_used),
        "latency_ms": round(result.latency_ms, 3),
        "arm_diagnostics": [asdict(d) for d in result.arm_diagnostics],
        "hits": [asdict(h) for h in result.hits],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.db_dsn or not args.db_dsn.strip():
        print("error: DATABASE_URL is missing", file=sys.stderr)
        return 4

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )

    embedder = LazyEmbedder(args.model_dir, args.max_seq_length, args.threads)

    try:
        filters = Filters(
            court=args.court,
            jurisdiction=args.jurisdiction,
            published_only=args.published_only,
            date_from=args.date_from,
            date_to=args.date_to,
            case_ids=args.case_id,
        )

        config = default_config()
        if args.config_override:
            overrides = dict(_parse_config_override(raw) for raw in args.config_override)
            config = config.with_overrides(**overrides)

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
        logger.error("invalid request: %s", exc)
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (psycopg.Error, OSError, RuntimeError) as exc:
        logger.exception("retrieval failed")
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("unexpected failure")
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        embedder.close()


if __name__ == "__main__":
    sys.exit(main())
