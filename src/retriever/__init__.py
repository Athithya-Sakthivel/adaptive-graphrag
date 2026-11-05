"""Public API for the retrieval package.

Exports the classifiers, config, filters, fusion primitives, router,
and result types. ``QueryEmbedder`` is exported lazily — the retriever
package must be importable without ``openvino`` for test-only installs
and for any consumer that only needs classification or fusion.

Import-time cost
----------------
``import retriever`` pulls in:
    * ``classify`` — re, enum
    * ``config``   — stdlib only
    * ``filters``  — psycopg types
    * ``fusion``   — numpy, dataclasses
    * ``router``   — numpy, psycopg, and (via ``from . import ...``)
                     dense, exact, graph, sparse

It does *not* pull in ``openvino``. ``QueryEmbedder`` is loaded on
first access via module-level ``__getattr__`` (PEP 562), which caches
into the module globals after the first call.

That means:
    * ``import retriever``                     — works without openvino
    * ``from retriever.classify import ...``   — works without openvino
    * ``from retriever import QueryEmbedder``  — triggers lazy load
    * ``retriever.QueryEmbedder``              — triggers lazy load

If ``openvino`` is not installed, only the lazy path raises
``ImportError``; everything else continues to work.
"""

from __future__ import annotations

# Add after existing imports:
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
from .fusion import Hit, case_dedup, rrf_pass1, rrf_pass2
from .router import ArmDiagnostic, RouterResult, route

if TYPE_CHECKING:
    from .embedder import QueryEmbedder

__all__ = [
    # classify
    "QueryClass",
    "classify",
    "extract_case_reference",
    "extract_citation",
    "extract_opinion_id",
    "has_case_reference",
    # config
    "RetrieverConfig",
    "default_config",
    "reset_default_config",
    # filters
    "Filters",
    # fusion
    "Hit",
    "case_dedup",
    "rrf_pass1",
    "rrf_pass2",
    # router
    "ArmDiagnostic",
    "RouterResult",
    "route",
    # embedder (lazy)
    "QueryEmbedder",
]


# ── Lazy exports ───────────────────────────────────────────────────────


def __getattr__(name: str) -> Any:
    """Lazy loader for ``QueryEmbedder`` (PEP 562).

    Only names in ``_LAZY_ATTRS`` are handled here; any other missing
    attribute raises ``AttributeError`` so typos are caught normally.
    After the first successful load the symbol is cached in module
    globals, so subsequent ``getattr`` calls hit the normal attribute
    path and never re-enter this function.
    """
    if name == "QueryEmbedder":
        from .embedder import QueryEmbedder

        globals()["QueryEmbedder"] = QueryEmbedder
        return QueryEmbedder

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Report only the public API to ``dir(retriever)``."""
    return sorted(__all__)
