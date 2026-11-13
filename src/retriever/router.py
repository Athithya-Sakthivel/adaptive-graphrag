"""Adaptive GraphRAG query routing.

Classifies queries, resolves target cases, orchestrates retrieval arms,
handles graceful degradation, and applies case diversity.

Route contracts:
    HOLDING  — dense + sparse. No graph, no case resolution.
    LOOKUP   — sparse + exact + 1-hop incoming graph. Never embeds.
    VALIDITY — sparse + exact + 1-hop graph, then dense restricted to
               the citing cases found by the graph.

Every arm failure is recorded in arm_diagnostics and the remaining arms
still run, so a degraded index yields partial results rather than an error.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from psycopg import Connection

from . import dense, exact, graph, sparse
from .classify import (
    QueryClass,
    classify,
    extract_case_reference,
    extract_citation,
    extract_opinion_id,
)
from .filters import Filters
from .fusion import Hit, case_dedup, rrf_pass1

if TYPE_CHECKING:
    from .config import RetrieverConfig

logger = logging.getLogger("retriever.router")

EMBED_CHAR_BUDGET = 350
EMBED_MAX_ATTEMPTS = 4
EXACT_CITATION_LIMIT = 20


@dataclass(frozen=True, slots=True)
class ArmDiagnostic:
    """Per-arm execution record for one route invocation."""

    name: str
    attempted: bool
    hits: int
    latency_ms: float
    error: str | None = None


@dataclass(frozen=True, slots=True)
class RouterResult:
    """Complete retrieval result with diagnostics."""

    query: str
    query_class: QueryClass
    arms_used: list[str]
    hits: list[Hit]
    latency_ms: float
    arm_diagnostics: list[ArmDiagnostic] = field(default_factory=list)


def _validate_top_k(top_k: int) -> None:
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError(f"top_k must be positive int, got {top_k!r}")


def _validate_query(query: str) -> str:
    if not isinstance(query, str):
        raise TypeError("query must be str")
    query = " ".join(query.split())
    if not query:
        raise ValueError("query must be non-empty")
    if len(query) > 10000:
        raise ValueError("query exceeds 10,000-character safety limit")
    return query


def _embed_query(embedder: Any, query: str) -> np.ndarray:
    """Encode a query, shrinking the text until it fits the model window.

    The compiled query model has a fixed 128-token window that includes
    the instruction prefix and special tokens. Passage-shaped queries
    can exceed it, and the embedder raises ValueError rather than
    truncating silently. Halving the character budget on that error keeps
    the dense arm available instead of losing it to a length overflow.
    """
    method = getattr(embedder, "embed_query", None)
    if not callable(method):
        raise TypeError("embedder must expose embed_query(text)")

    text = query[:EMBED_CHAR_BUDGET]
    vector: np.ndarray | None = None

    for _ in range(EMBED_MAX_ATTEMPTS):
        if not text.strip():
            break
        try:
            vector = np.asarray(method(text), dtype=np.float32)
            break
        except ValueError:
            logger.debug("query exceeds model window at %d chars; halving", len(text))
            text = text[: len(text) // 2]

    if vector is None:
        raise RuntimeError("query could not be embedded within the model sequence limit")

    if vector.ndim != 1 or not np.isfinite(vector).all():
        raise RuntimeError(f"embedder returned invalid vector: shape={vector.shape}")

    if float(np.linalg.norm(vector)) <= 1e-12:
        raise RuntimeError("embedder returned zero vector")

    return vector


def _run_arm[T](
    conn: Connection | None,
    arm_name: str,
    operation: Callable[[], list[T]],
    diagnostics: list[ArmDiagnostic],
) -> list[T]:
    """Execute one arm, record its diagnostic, return [] on failure.

    Each arm runs inside its own transaction so a failure does not poison
    the connection for subsequent arms.
    """
    start = time.monotonic()
    try:
        if conn is None:
            results = operation()
        else:
            with conn.transaction():
                results = operation()
        latency_ms = (time.monotonic() - start) * 1000
        diagnostics.append(ArmDiagnostic(arm_name, True, len(results), latency_ms, None))
        return results
    except Exception as exc:
        latency_ms = (time.monotonic() - start) * 1000
        logger.exception("%s retrieval arm failed; continuing with other arms", arm_name)
        diagnostics.append(
            ArmDiagnostic(arm_name, True, 0, latency_ms, f"{type(exc).__name__}: {exc}")
        )
        return []


def _resolve_target_ids(
    conn: Connection | None,
    query: str,
    citation_hits: list[Hit],
    diagnostics: list[ArmDiagnostic],
    config: RetrieverConfig,
) -> list[int]:
    """Resolve the named legal authority before graph traversal.

    Priority: explicit opinion ID, then citation hits, then case-name
    match (exact first, trigram fallback). Returns [] when nothing
    resolves, which causes callers to skip graph traversal entirely
    rather than seeding it with arbitrary cases.
    """
    explicit_id = extract_opinion_id(query)
    if explicit_id is not None:
        ids = _run_arm(
            conn,
            "opinion_id_resolution",
            lambda: exact.resolve_opinion_id(conn, explicit_id),
            diagnostics,
        )
        return list(dict.fromkeys(int(cid) for cid in ids))

    if citation_hits:
        seen: list[int] = []
        for hit in citation_hits:
            if hit.case_id not in seen:
                seen.append(hit.case_id)
                if len(seen) >= config.lookup_seed_limit:
                    break
        return seen

    case_name = extract_case_reference(query)
    if not case_name:
        return []

    ids = _run_arm(
        conn,
        "case_name_resolution",
        lambda: exact.resolve_case_ids(
            conn, case_name, limit=config.lookup_seed_limit, config=config
        ),
        diagnostics,
    )
    return list(dict.fromkeys(int(cid) for cid in ids))


def _expand_parents(
    conn: Connection | None,
    case_ids: list[int],
    filters: Filters,
    arm_name: str,
    diagnostics: list[ArmDiagnostic],
) -> list[Hit]:
    if not case_ids:
        return []
    return _run_arm(
        conn,
        arm_name,
        lambda: exact.case_ids_to_hits(conn, case_ids, filters=filters),
        diagnostics,
    )


def _search_citers(
    conn: Connection | None,
    resolved_ids: list[int],
    filters: Filters,
    diagnostics: list[ArmDiagnostic],
    config: RetrieverConfig,
    *,
    direction: graph.Direction,
    cap_per_hop: int,
) -> list[dict]:
    return _run_arm(
        conn,
        "graph_1hop",
        lambda: graph.search(
            conn,
            resolved_ids,
            direction=direction,
            depth=1,
            cap_per_hop=cap_per_hop,
            limit=config.graph_limit,
            filters=filters,
            config=config,
        ),
        diagnostics,
    )


def _graph_hits(
    conn: Connection | None,
    graph_results: list[dict],
    filters: Filters,
    diagnostics: list[ArmDiagnostic],
    direction: str,
) -> list[Hit]:
    """Expand graph case IDs to parent hits, preserving hop provenance."""
    if not graph_results:
        return []

    case_ids = [r["case_id"] for r in graph_results]
    hits = _expand_parents(conn, case_ids, filters, "graph_parent_expansion", diagnostics)

    hop_by_case: dict[int, int] = {}
    for row in graph_results:
        cid = row["case_id"]
        hop = row["hop"]
        prev = hop_by_case.get(cid)
        if prev is None or hop < prev:
            hop_by_case[cid] = hop

    return [
        Hit(
            chunk_id=h.chunk_id,
            case_id=h.case_id,
            is_parent=h.is_parent,
            score=float(hop_by_case.get(h.case_id, 1)),
            rank=0,
            meta={
                **h.meta,
                "hop": hop_by_case.get(h.case_id, 1),
                "graph_direction": direction,
                "retrieval_arm": "graph",
            },
        )
        for h in hits
    ]


def _citation_hits(
    conn: Connection | None,
    query: str,
    filters: Filters,
    diagnostics: list[ArmDiagnostic],
    config: RetrieverConfig,
    arms_used: list[str],
) -> list[Hit]:
    """Run the exact arm when the query carries a reporter citation."""
    citation = extract_citation(query)
    if not citation:
        return []
    arms_used.append("exact")
    return _run_arm(
        conn,
        "exact",
        lambda: exact.search(
            conn, citation, limit=EXACT_CITATION_LIMIT, filters=filters, config=config
        ),
        diagnostics,
    )


def _apply_case_dedup(
    hits: list[Hit], query_class: QueryClass, config: RetrieverConfig
) -> list[Hit]:
    if not config.case_diversity_enabled:
        return hits
    return case_dedup(hits, max_per_case=config.case_max_per_class(query_class.value))


def _finish(
    *,
    query: str,
    query_class: QueryClass,
    arms_used: list[str],
    arms: dict[str, list[Hit]],
    k_values: dict[str, int],
    top_k: int,
    config: RetrieverConfig,
    diagnostics: list[ArmDiagnostic],
    started: float,
) -> RouterResult:
    """Fuse, enforce case diversity, truncate, and package the result."""
    fused = rrf_pass1(arms, k_values, limit=top_k * config.rrf_output_multiplier)
    fused = _apply_case_dedup(fused, query_class, config)

    logger.info(
        "%s %s",
        query_class.value,
        " ".join(f"{name}={len(hits)}" for name, hits in arms.items()),
    )

    return RouterResult(
        query=query,
        query_class=query_class,
        arms_used=arms_used,
        hits=fused[:top_k],
        latency_ms=(time.monotonic() - started) * 1000,
        arm_diagnostics=diagnostics,
    )


def _route_holding(
    conn: Connection | None,
    embedder: Any,
    query: str,
    filters: Filters,
    top_k: int,
    config: RetrieverConfig,
    started: float,
) -> RouterResult:
    """Semantic passage retrieval: dense + sparse, no graph.

    The exact arm is intentionally absent. classify() routes any query
    containing a reporter citation to LOOKUP, so a HOLDING query never
    carries one.
    """
    diagnostics: list[ArmDiagnostic] = []
    arms_used: list[str] = []
    dense_hits: list[Hit] = []

    try:
        query_vec = _embed_query(embedder, query)
        arms_used.append("dense")
        dense_hits = _run_arm(
            conn,
            "dense",
            lambda: dense.search(
                conn, query_vec, filters=filters, limit=config.dense_limit, config=config
            ),
            diagnostics,
        )
    except Exception:
        logger.exception("HOLDING dense arm unavailable; retaining lexical retrieval")

    arms_used.append("sparse")
    sparse_hits = _run_arm(
        conn,
        "sparse",
        lambda: sparse.search(conn, query, filters=filters, limit=config.sparse_limit),
        diagnostics,
    )

    return _finish(
        query=query,
        query_class=QueryClass.HOLDING,
        arms_used=arms_used,
        arms={"dense": dense_hits, "sparse": sparse_hits},
        k_values=config.holding_rrf_k(),
        top_k=top_k,
        config=config,
        diagnostics=diagnostics,
        started=started,
    )


def _route_lookup(
    conn: Connection | None,
    query: str,
    filters: Filters,
    top_k: int,
    config: RetrieverConfig,
    started: float,
) -> RouterResult:
    """Citation lookup: resolve the target, then traverse incoming edges.

    Never touches the embedder — citation lookup is structural, so the
    model load cost is avoided entirely on this path.
    """
    diagnostics: list[ArmDiagnostic] = []
    arms_used: list[str] = ["sparse"]

    sparse_hits = _run_arm(
        conn,
        "sparse",
        lambda: sparse.search(conn, query, filters=filters, limit=config.sparse_limit),
        diagnostics,
    )

    exact_hits = _citation_hits(conn, query, filters, diagnostics, config, arms_used)
    resolved_ids = _resolve_target_ids(conn, query, exact_hits, diagnostics, config)

    resolved_hits: list[Hit] = []
    if not exact_hits and resolved_ids:
        resolved_hits = _expand_parents(
            conn, resolved_ids, filters, "resolved_case_parents", diagnostics
        )
        if resolved_hits:
            arms_used.append("case_resolution")

    graph_hits: list[Hit] = []
    if resolved_ids:
        arms_used.append("graph_1hop")
        graph_results = _search_citers(
            conn,
            resolved_ids,
            filters,
            diagnostics,
            config,
            direction=config.lookup_graph_direction,
            cap_per_hop=config.lookup_graph_cap_per_hop,
        )
        graph_hits = _graph_hits(
            conn, graph_results, filters, diagnostics, config.lookup_graph_direction
        )
    else:
        logger.warning("LOOKUP graph skipped: target case could not be resolved")

    return _finish(
        query=query,
        query_class=QueryClass.LOOKUP,
        arms_used=arms_used,
        arms={
            "sparse": sparse_hits,
            "exact": exact_hits + resolved_hits,
            "graph_1hop": graph_hits,
        },
        k_values=config.lookup_rrf_k(),
        top_k=top_k,
        config=config,
        diagnostics=diagnostics,
        started=started,
    )


def _route_validity(
    conn: Connection | None,
    embedder: Any,
    query: str,
    filters: Filters,
    top_k: int,
    config: RetrieverConfig,
    started: float,
) -> RouterResult:
    """Overruling detection: graph finds citers, dense searches within them.

    Raw CITES edges carry no treatment labels, so overruling evidence is
    located by running dense retrieval restricted to the citing cases.
    This is a runtime approximation of an offline treatment classifier.
    """
    diagnostics: list[ArmDiagnostic] = []
    arms_used: list[str] = ["sparse"]

    sparse_hits = _run_arm(
        conn,
        "sparse",
        lambda: sparse.search(conn, query, filters=filters, limit=config.sparse_limit),
        diagnostics,
    )

    exact_hits = _citation_hits(conn, query, filters, diagnostics, config, arms_used)
    resolved_ids = _resolve_target_ids(conn, query, exact_hits, diagnostics, config)

    graph_hits: list[Hit] = []
    dense_citer_hits: list[Hit] = []

    if not resolved_ids:
        logger.warning("VALIDITY graph skipped: target case could not be resolved")
    else:
        arms_used.append("graph_1hop")
        graph_results = _search_citers(
            conn,
            resolved_ids,
            filters,
            diagnostics,
            config,
            direction=config.validity_graph_direction,
            cap_per_hop=config.validity_graph_cap_per_hop,
        )
        graph_hits = _graph_hits(
            conn, graph_results, filters, diagnostics, config.validity_graph_direction
        )

        if graph_results:
            graph_case_ids = list(dict.fromkeys(r["case_id"] for r in graph_results))[
                : config.graph_limit
            ]
            citer_filters = filters.with_case_ids(graph_case_ids)
            if not citer_filters.matches_nothing:
                arms_used.append("dense_citers")
                try:
                    query_vec = _embed_query(embedder, query)
                    dense_citer_hits = _run_arm(
                        conn,
                        "dense_citers",
                        lambda: dense.search(
                            conn,
                            query_vec,
                            filters=citer_filters,
                            limit=config.validity_citers_limit,
                            config=config,
                        ),
                        diagnostics,
                    )
                except Exception:
                    logger.exception(
                        "VALIDITY dense_citers unavailable; retaining sparse and graph evidence"
                    )

    return _finish(
        query=query,
        query_class=QueryClass.VALIDITY,
        arms_used=arms_used,
        arms={
            "sparse": sparse_hits,
            "exact": exact_hits,
            "graph_1hop": graph_hits,
            "dense_citers": dense_citer_hits,
        },
        k_values=config.validity_rrf_k(),
        top_k=top_k,
        config=config,
        diagnostics=diagnostics,
        started=started,
    )


def route(
    conn: Connection | None,
    embedder: Any,
    query: str,
    *,
    filters: Filters | None = None,
    top_k: int = 10,
    config: RetrieverConfig | None = None,
) -> RouterResult:
    """Classify and execute the appropriate retrieval route.

    Args:
        conn: Open psycopg connection, or None for fully mocked retrieval.
        embedder: Object exposing embed_query(text). Untouched on LOOKUP.
        query: Natural-language legal question.
        filters: Metadata restrictions applied inside every arm's SQL.
        top_k: Maximum hits to return after fusion and deduplication.
        config: Retrieval configuration; defaults to default_config().

    Raises:
        TypeError: If query is not a string or filters is not Filters.
        ValueError: If query is blank or top_k is not a positive int.
    """
    normalized_query = _validate_query(query)
    _validate_top_k(top_k)

    if filters is None:
        filters = Filters()
    if not isinstance(filters, Filters):
        raise TypeError("filters must be Filters or None")

    if config is None:
        from .config import default_config

        config = default_config()

    query_class = classify(normalized_query)

    if filters.matches_nothing:
        logger.info("route skipped: explicit empty case-ID filter")
        return RouterResult(
            query=normalized_query,
            query_class=query_class,
            arms_used=[],
            hits=[],
            latency_ms=0.0,
            arm_diagnostics=[],
        )

    started = time.monotonic()
    logger.info("query_class=%s route starting", query_class.value)

    if query_class is QueryClass.HOLDING:
        return _route_holding(conn, embedder, normalized_query, filters, top_k, config, started)
    if query_class is QueryClass.LOOKUP:
        return _route_lookup(conn, normalized_query, filters, top_k, config, started)
    if query_class is QueryClass.VALIDITY:
        return _route_validity(conn, embedder, normalized_query, filters, top_k, config, started)

    raise RuntimeError(f"unhandled query class: {query_class!r}")


__all__ = [
    "ArmDiagnostic",
    "RouterResult",
    "route",
]
