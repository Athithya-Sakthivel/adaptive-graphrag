"""Adaptive GraphRAG retrieval system for US case law.

Public API exports. QueryEmbedder is lazy-loaded to avoid pulling
openvino into test-only environments.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .classify import (
    QueryClass,
    classify,
    extract_case_reference,
    extract_citation,
    extract_opinion_id,
    has_case_reference,
)
from .config import RetrieverConfig, default_config, reset_default_config
from .filters import Filters
from .fusion import Hit, case_dedup, rrf_pass1
from .router import ArmDiagnostic, RouterResult, route

if TYPE_CHECKING:
    from .embedder import QueryEmbedder

__all__ = [
    "QueryClass",
    "classify",
    "extract_case_reference",
    "extract_citation",
    "extract_opinion_id",
    "has_case_reference",
    "RetrieverConfig",
    "default_config",
    "reset_default_config",
    "Filters",
    "Hit",
    "case_dedup",
    "rrf_pass1",
    "ArmDiagnostic",
    "RouterResult",
    "route",
    "QueryEmbedder",
]

_LAZY_ATTRS = {"QueryEmbedder"}


def __getattr__(name: str) -> Any:
    if name == "QueryEmbedder":
        from .embedder import QueryEmbedder as _QE

        globals()["QueryEmbedder"] = _QE
        return _QE
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
