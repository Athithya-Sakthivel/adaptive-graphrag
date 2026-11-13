"""Adaptive GraphRAG query routing.

Classifies queries, resolves target cases, orchestrates retrieval
arms, handles graceful degradation, and applies case diversity.
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
    if not isinstance(top_k, int) or top_k < 1:
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
    """Encode query using instruction-prefixed embedder."""
    # Truncate to 350 chars to prevent exceeding the 128-token model limit
    query = query[:350]

    method = getattr(embedder, "embed_query", None)
    if not callable(method):
        raise TypeError("embedder must expose embed_query(text)")

    vector = np.asarray(method(query), dtype=np.float32)
    if vector.ndim != 1 or not np.isfinite(vector).all():
        raise RuntimeError(f"embedder returned invalid vector: shape={vector.shape}")

    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise RuntimeError("embedder returned zero vector")
    return vector


def _run_arm(
    conn: Connection | None,
    arm_name: str,
    operation: Callable[[], list[Any]],
    diagnostics: list[ArmDiagnostic],
) -> list[Any]:
    """Execute one arm, record diagnostic, return results (or [] on failure)."""
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
    match (exact first, trigram fallback).
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


def _graph_hits(
    conn: Connection | None,
    graph_results: list[dict],
    filters: Filters,
    diagnostics: list[ArmDiagnostic],
    direction: str,
) -> list[Hit]:
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


def _apply_case_dedup(
    hits: list[Hit], query_class: QueryClass, config: RetrieverConfig
) -> list[Hit]:
    if not config.case_diversity_enabled:
        return hits
    return case_dedup(hits, max_per_case=config.case_max_per_class(query_class.value))


def _route_holding(conn, embedder, query, filters, top_k, config, started):
    diagnostics: list[ArmDiagnostic] = []
    arms_used: list[str] = []

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
        dense_hits = []

    arms_used.append("sparse")
    sparse_hits = _run_arm(
        conn,
        "sparse",
        lambda: sparse.search(conn, query, filters=filters, limit=config.sparse_limit),
        diagnostics,
    )

    citation = extract_citation(query)
    exact_hits: list[Hit] = []
    if citation:
        arms_used.append("exact")
        exact_hits = _run_arm(
            conn,
            "exact",
            lambda: exact.search(conn, citation, limit=20, filters=filters, config=config),
            diagnostics,
        )

    fused = rrf_pass1(
        {"dense": dense_hits, "sparse": sparse_hits, "exact": exact_hits},
        config.holding_rrf_k(),
        limit=top_k * config.rrf_output_multiplier,
    )
    fused = _apply_case_dedup(fused, QueryClass.HOLDING, config)

    logger.info(
        "HOLDING dense=%d sparse=%d exact=%d", len(dense_hits), len(sparse_hits), len(exact_hits)
    )

    return RouterResult(
        query=query,
        query_class=QueryClass.HOLDING,
        arms_used=arms_used,
        hits=fused[:top_k],
        latency_ms=(time.monotonic() - started) * 1000,
        arm_diagnostics=diagnostics,
    )


def _route_lookup(conn, query, filters, top_k, config, started):
    diagnostics: list[ArmDiagnostic] = []
    arms_used: list[str] = ["sparse"]

    sparse_hits = _run_arm(
        conn,
        "sparse",
        lambda: sparse.search(conn, query, filters=filters, limit=config.sparse_limit),
        diagnostics,
    )

    citation = extract_citation(query)
    exact_hits: list[Hit] = []
    if citation:
        arms_used.append("exact")
        exact_hits = _run_arm(
            conn,
            "exact",
            lambda: exact.search(conn, citation, limit=20, filters=filters, config=config),
            diagnostics,
        )

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
        graph_results = _run_arm(
            conn,
            "graph_1hop",
            lambda: graph.search(
                conn,
                resolved_ids,
                direction=config.lookup_graph_direction,
                depth=1,
                cap_per_hop=config.lookup_graph_cap_per_hop,
                limit=config.graph_limit,
                filters=filters,
                config=config,
            ),
            diagnostics,
        )
        graph_hits = _graph_hits(
            conn, graph_results, filters, diagnostics, config.lookup_graph_direction
        )
    else:
        logger.warning("LOOKUP graph skipped: target case could not be resolved")

    fused = rrf_pass1(
        {"sparse": sparse_hits, "exact": exact_hits + resolved_hits, "graph_1hop": graph_hits},
        config.lookup_rrf_k(),
        limit=top_k * config.rrf_output_multiplier,
    )
    fused = _apply_case_dedup(fused, QueryClass.LOOKUP, config)

    logger.info(
        "LOOKUP sparse=%d exact=%d case_resolution=%d graph_1hop=%d",
        len(sparse_hits),
        len(exact_hits),
        len(resolved_hits),
        len(graph_hits),
    )

    return RouterResult(
        query=query,
        query_class=QueryClass.LOOKUP,
        arms_used=arms_used,
        hits=fused[:top_k],
        latency_ms=(time.monotonic() - started) * 1000,
        arm_diagnostics=diagnostics,
    )


def _route_validity(conn, embedder, query, filters, top_k, config, started):
    diagnostics: list[ArmDiagnostic] = []
    arms_used: list[str] = ["sparse"]

    sparse_hits = _run_arm(
        conn,
        "sparse",
        lambda: sparse.search(conn, query, filters=filters, limit=config.sparse_limit),
        diagnostics,
    )

    citation = extract_citation(query)
    exact_hits: list[Hit] = []
    if citation:
        arms_used.append("exact")
        exact_hits = _run_arm(
            conn,
            "exact",
            lambda: exact.search(conn, citation, limit=20, filters=filters, config=config),
            diagnostics,
        )

    resolved_ids = _resolve_target_ids(conn, query, exact_hits, diagnostics, config)

    if not resolved_ids:
        logger.warning("VALIDITY graph skipped: target case could not be resolved")
        fused = rrf_pass1(
            {"sparse": sparse_hits, "exact": exact_hits},
            {"sparse": config.sparse_k_validity, "exact": config.exact_k},
            limit=top_k * config.rrf_output_multiplier,
        )
        fused = _apply_case_dedup(fused, QueryClass.VALIDITY, config)
        return RouterResult(
            query=query,
            query_class=QueryClass.VALIDITY,
            arms_used=arms_used,
            hits=fused[:top_k],
            latency_ms=(time.monotonic() - started) * 1000,
            arm_diagnostics=diagnostics,
        )

    arms_used.append("graph_1hop")
    graph_results = _run_arm(
        conn,
        "graph_1hop",
        lambda: graph.search(
            conn,
            resolved_ids,
            direction=config.validity_graph_direction,
            depth=1,
            cap_per_hop=config.validity_graph_cap_per_hop,
            limit=config.graph_limit,
            filters=filters,
            config=config,
        ),
        diagnostics,
    )
    graph_hits = _graph_hits(
        conn, graph_results, filters, diagnostics, config.validity_graph_direction
    )

    dense_citer_hits: list[Hit] = []
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

    fused = rrf_pass1(
        {
            "sparse": sparse_hits,
            "exact": exact_hits,
            "graph_1hop": graph_hits,
            "dense_citers": dense_citer_hits,
        },
        config.validity_rrf_k(),
        limit=top_k * config.rrf_output_multiplier,
    )
    fused = _apply_case_dedup(fused, QueryClass.VALIDITY, config)

    logger.info(
        "VALIDITY sparse=%d exact=%d graph_1hop=%d dense_citers=%d",
        len(sparse_hits),
        len(exact_hits),
        len(graph_hits),
        len(dense_citer_hits),
    )

    return RouterResult(
        query=query,
        query_class=QueryClass.VALIDITY,
        arms_used=arms_used,
        hits=fused[:top_k],
        latency_ms=(time.monotonic() - started) * 1000,
        arm_diagnostics=diagnostics,
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
    """Classify and execute the appropriate retrieval route."""
    normalized_query = _validate_query(query)
    _validate_top_k(top_k)

    if filters is None:
        filters = Filters()
    if not isinstance(filters, Filters):
        raise TypeError("filters must be Filters or None")

    if config is None:
        from .config import default_config

        config = default_config()

    if filters.matches_nothing:
        logger.info("route skipped: explicit empty case-ID filter")
        return RouterResult(
            query=normalized_query,
            query_class=classify(normalized_query),
            arms_used=[],
            hits=[],
            latency_ms=0.0,
            arm_diagnostics=[],
        )

    started = time.monotonic()
    query_class = classify(normalized_query)
    logger.info("query_class=%s route starting", query_class.value)

    if query_class == QueryClass.HOLDING:
        return _route_holding(conn, embedder, normalized_query, filters, top_k, config, started)
    if query_class == QueryClass.LOOKUP:
        return _route_lookup(conn, normalized_query, filters, top_k, config, started)
    if query_class == QueryClass.VALIDITY:
        return _route_validity(conn, embedder, normalized_query, filters, top_k, config, started)

    raise RuntimeError(f"unhandled query class: {query_class!r}")


__all__ = [
    "ArmDiagnostic",
    "RouterResult",
    "route",
]
