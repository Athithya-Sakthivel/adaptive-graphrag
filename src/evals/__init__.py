"""Evaluation framework for the Adaptive GraphRAG retriever.

Provides gold-set generation, corpus fingerprinting, retrieval adapters,
and metric computation with bootstrap confidence intervals.

Imports are lazy at runtime so that importing this package does not
pull in psycopg, openvino, or the retriever until a symbol is used.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .adapter import EvalRuntime, build_runtime
    from .fingerprint import case_law_fingerprint
    from .source import EvalSource, build_source

__all__ = [
    "EvalRuntime",
    "EvalSource",
    "build_runtime",
    "build_source",
    "case_law_fingerprint",
]


def __getattr__(name: str) -> Any:
    if name == "case_law_fingerprint":
        from .fingerprint import case_law_fingerprint as _fn

        return _fn
    if name in ("build_source", "EvalSource"):
        from . import source as _source

        return getattr(_source, name)
    if name in ("build_runtime", "EvalRuntime"):
        from . import adapter as _adapter

        return getattr(_adapter, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
