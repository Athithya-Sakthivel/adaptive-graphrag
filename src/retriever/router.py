"""Adaptive GraphRAG query routing.

This module owns:
    - Query classification (delegated to classify.py)
    - Target case resolution
    - Retrieval-arm selection
    - Graceful arm failure handling
    - One-pass and two-pass RRF orchestration
    - Per-arm latency and error diagnostics
    - Route-level case diversity enforcement

It does NOT own:
    - Embedding model inference implementation
    - SQL implementation of individual arms
    - Parent-context assembly
    - LLM generation
    - Citation verification

Config wiring
-------------
Every tunable — per-arm RRF k, per-arm limits, HNSW params, trigram
threshold, case-diversity caps, APPLICATION graph direction and
depth, 2-hop gate — lives in ``config: RetrieverConfig``. When
``config`` is None, ``default_config()`` (which reads ``RETRIEVER_*``
env vars on first call) is used. Precedence:

    explicit ``config`` → ``default_config()`` → env vars → defaults

APPLICATION direction (audit-fix)
---------------------------------
Pre-hardening APPLICATION used ``direction="outgoing"``: "how has X
been applied" returned cases that X cites. The audit flagged this
as P0 — the semantic target is cases that CITE X (applications of
X), not cases X cites. Direction is now
``config.application_graph_direction`` (default ``"incoming"``).

Seed source
-----------
APPLICATION seeds from the *resolved case reference*, not from
pass-1 dense+sparse. For "how has Roe v. Wade been applied":
    * ``extract_case_reference`` → "Roe v. Wade"
    * ``exact.resolve_case_ids`` (with trigram fallback) → [case_id]
    * ``graph.search([case_id], direction="incoming")`` → citers

The target case is filtered out of pass-1 before ``rrf_pass2`` so
that results are applications of X, not X itself. LOOKUP and
VALIDITY seed sources are unchanged — they already resolve to the
target case before graph traversal.

Doctrine-APPLICATION fallback
-----------------------------
When ``classify()`` returns APPLICATION but no case reference can
be resolved (e.g. "how has qualified immunity been applied"), the
route falls back to ``dense + sparse`` only — no graph seed exists.
The three distinct cases are logged separately:

    * No case name in query         → info: "doctrine-only, graph skipped"
    * Case name present, unresolved → warning: reference not in corpus
    * Resolved, zero citers         → warning: graph returned no citers

Case diversity (fusion post-filter)
-----------------------------------
When ``config.case_diversity_enabled`` is True, ``case_dedup`` is
applied after RRF pass 1 / pass 2 with
``config.case_max_per_class(query_class.value)``. Fusion requests
``top_k * config.rrf_output_multiplier`` entries so dedup has
headroom; final output is truncated to ``top_k``.

Per-arm diagnostics
-------------------
Every arm invocation produces an ``ArmDiagnostic`` entry, including
resolution helpers (``opinion_id_resolution``, ``case_name_resolution``,
``exact``, ``resolved_case_parents``, ``graph_parent_expansion``).
``RouterResult.arm_diagnostics`` exposes all entries in execution
order for full latency attribution.

``arms_used`` vs ``arm_diagnostics``
------------------------------------
``arms_used`` lists arms that contribute to the fusion stage
(dense, sparse, exact, graph_1hop, dense_citers, graph_2hop, and
``case_resolution`` where applicable). Resolution helpers such as
citation search in APPLICATION do not contribute to fusion and are
therefore not listed in ``arms_used``, even though they appear in
``arm_diagnostics``.

2-hop gate
----------
``config.application_2hop_enabled`` defaults to False and must not
be flipped until the documented ablation shows ≥5 points R@10 gain
over 1-hop on multi-hop queries.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
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
from .fusion import Hit, case_dedup, rrf_pass1, rrf_pass2

if TYPE_CHECKING:
    from .config import RetrieverConfig


logger = logging.getLogger("retriever.router")

EMBEDDING_DIMENSIONS = 1024

# Deprecated module constant retained for external imports. The
# authoritative value lives in config.application_2hop_enabled.
APPLICATION_2HOP_ENABLED = False


# ── Result types ───────────────────────────────────────────────────────


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
    query: str
    query_class: QueryClass
    arms_used: list[str]
    hits: list[Hit]
    latency_ms: float
    arm_diagnostics: list[ArmDiagnostic] = field(default_factory=list)


# ── Validation ─────────────────────────────────────────────────────────


def _validate_top_k(top_k: int) -> None:
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer")


def _validate_query(query: str) -> str:
    if not isinstance(query, str):
        raise TypeError("query must be a string")

    query = " ".join(query.split())
    if not query:
        raise ValueError("query must be a non-empty string")
    if len(query) > 10000:
        raise ValueError("query exceeds the 10,000-character safety limit")

    return query


# ── Query embedding ────────────────────────────────────────────────────


def _embed_query(embedder: Any, query: str) -> np.ndarray:
    """Encode a query using the query-specific instruction prefix.

    Uses the configured tokenizer to pre-check length; no silent
    truncation. Applies to ``QueryEmbedder`` (retriever.embedder) and
    any duck-typed embedder exposing ``embed_query``, ``tokenizer``,
    and ``max_seq_length``.
    """
    method = getattr(embedder, "embed_query", None)
    if not callable(method):
        raise TypeError("embedder must expose embed_query(text)")

    tokenizer = getattr(embedder, "tokenizer", None)
    if tokenizer is None:
        model = getattr(embedder, "model", None)
        tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError(
            "selected embedder does not expose its tokenizer; "
            "query-length validation cannot be performed safely"
        )

    # Prefer retriever.embedder's QUERY_PREFIX; fall back to a
    # self-contained copy if openvino is not importable (test-only
    # installs). The string is identical.
    try:
        from .embedder import QUERY_PREFIX
    except ImportError:  # pragma: no cover - test-only fallback
        QUERY_PREFIX = (
            "Instruct: Given a web search query, retrieve relevant "
            "passages that answer the query\nQuery: "
        )

    configured_limit = getattr(embedder, "max_seq_length", None)
    if configured_limit is None:
        raise RuntimeError("embedder does not expose max_seq_length")
    configured_limit = int(configured_limit)

    fixed_sequence = getattr(embedder, "fixed_sequence_length", None)
    if fixed_sequence is not None:
        configured_limit = min(configured_limit, int(fixed_sequence))
    elif hasattr(embedder, "compiled_model"):
        try:
            from indexing_pipeline.embedder import STATIC_SEQUENCE

            configured_limit = min(configured_limit, int(STATIC_SEQUENCE))
        except ImportError:  # pragma: no cover - defensive
            pass

    if configured_limit < 1:
        raise RuntimeError("embedder reports an invalid sequence limit")

    token_count = len(
        tokenizer.encode(
            QUERY_PREFIX + query,
            add_special_tokens=True,
            truncation=False,
        )
    )
    if token_count > configured_limit:
        raise ValueError(
            f"prefixed query requires {token_count} tokens, "
            f"but this backend permits {configured_limit}. "
            "Shorten the query; automatic truncation is disabled."
        )

    vector = np.asarray(method(query), dtype=np.float32)
    if vector.shape != (EMBEDDING_DIMENSIONS,):
        raise RuntimeError(
            f"embedder returned shape {vector.shape}; expected ({EMBEDDING_DIMENSIONS},)"
        )
    if not np.isfinite(vector).all():
        raise RuntimeError("embedder returned NaN or infinite values")

    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise RuntimeError("embedder returned a zero query vector")

    return vector


# ── Arm execution with diagnostics ────────────────────────────────────
def _run_arm[T](
    conn: Connection | None,
    arm_name: str,
    operation: Callable[[], list[T]],
) -> tuple[list[T], bool, float, str | None]:
    """Execute one arm in an isolated transaction boundary.

    Returns ``(results, attempted, latency_ms, error)``. The savepoint
    rolls back on failure so one arm's exception cannot abort the
    outer transaction. ``conn is None`` is supported for unit tests
    with mocked arms.
    """
    start = time.monotonic()
    try:
        if conn is None:
            results = operation()
        else:
            with conn.transaction():
                results = operation()
        latency_ms = (time.monotonic() - start) * 1000
        return results, True, latency_ms, None
    except Exception as exc:
        latency_ms = (time.monotonic() - start) * 1000
        logger.exception(
            "%s retrieval arm failed; continuing with other available arms",
            arm_name,
        )
        return [], True, latency_ms, f"{type(exc).__name__}: {exc}"


def _run_and_record[T](
    conn: Connection | None,
    arm_name: str,
    operation: Callable[[], list[T]],
    diagnostics: list[ArmDiagnostic],
) -> list[T]:
    """Run one arm, append its diagnostic, return the results."""
    results, attempted, latency_ms, error = _run_arm(conn, arm_name, operation)
    diagnostics.append(
        ArmDiagnostic(
            name=arm_name,
            attempted=attempted,
            hits=len(results),
            latency_ms=latency_ms,
            error=error,
        )
    )
    return results


# ── Seed and expansion helpers ─────────────────────────────────────────


def _top_case_ids(hits: list[Hit], limit: int) -> list[int]:
    """Extract distinct case IDs in retrieval rank order."""
    if limit < 1:
        return []

    output: list[int] = []
    seen: set[int] = set()
    for hit in hits:
        if hit.case_id in seen:
            continue
        output.append(hit.case_id)
        seen.add(hit.case_id)
        if len(output) >= limit:
            break
    return output


def _resolve_target_ids(
    conn: Connection | None,
    query: str,
    citation_hits: list[Hit],
    diagnostics: list[ArmDiagnostic],
    config: RetrieverConfig,
) -> list[int]:
    """Resolve the named legal authority before graph traversal.

    Priority:
        0. Explicit CourtListener opinion ID in the query
        1. Exact reporter-citation matches from previous exact.search()
        2. Case-name match (exact first, trigram fallback via
           exact.resolve_case_ids)

    Returns [] if no resolution succeeded. The caller must fail closed
    (skip graph traversal) rather than use semantically similar
    neighbors as a substitute for identity.
    """
    explicit_id = extract_opinion_id(query)
    if explicit_id is not None:
        ids = _run_and_record(
            conn,
            "opinion_id_resolution",
            lambda: exact.resolve_opinion_id(conn, explicit_id),
            diagnostics,
        )
        return list(dict.fromkeys(int(case_id) for case_id in ids))

    citation_ids = _top_case_ids(citation_hits, config.lookup_seed_limit)
    if citation_ids:
        return citation_ids

    case_name = extract_case_reference(query)
    if not case_name:
        return []

    ids = _run_and_record(
        conn,
        "case_name_resolution",
        lambda: exact.resolve_case_ids(
            conn,
            case_name,
            limit=config.lookup_seed_limit,
            config=config,
        ),
        diagnostics,
    )
    return list(dict.fromkeys(int(case_id) for case_id in ids))


def _resolution_hits(
    conn: Connection | None,
    case_ids: list[int],
    filters: Filters,
    diagnostics: list[ArmDiagnostic],
    config: RetrieverConfig,
) -> list[Hit]:
    """Materialize parent Hits for case-name-resolved targets."""
    if not case_ids:
        return []

    hits = _run_and_record(
        conn,
        "resolved_case_parents",
        lambda: graph.case_ids_to_hits(conn, case_ids, filters=filters, config=config),
        diagnostics,
    )

    return [
        replace(
            hit,
            meta={**hit.meta, "retrieval_arm": "case_resolution"},
        )
        for hit in hits
    ]


def _graph_hits(
    conn: Connection | None,
    graph_results: list[dict[str, Any]],
    filters: Filters,
    diagnostics: list[ArmDiagnostic],
    config: RetrieverConfig,
) -> list[Hit]:
    """Expand graph case IDs into parent Hits with hop preserved."""
    if not graph_results:
        return []

    hop_by_case: dict[int, int] = {}
    for row in graph_results:
        case_id = row.get("case_id")
        hop = row.get("hop")
        if isinstance(case_id, bool) or not isinstance(case_id, int):
            raise RuntimeError(f"invalid graph case_id: {case_id!r}")
        if isinstance(hop, bool) or not isinstance(hop, int) or hop < 1:
            raise RuntimeError(f"invalid graph hop: {hop!r}")
        previous = hop_by_case.get(case_id)
        if previous is None or hop < previous:
            hop_by_case[case_id] = hop

    ids = sorted(
        hop_by_case,
        key=lambda case_id: (hop_by_case[case_id], case_id),
    )

    hits = _run_and_record(
        conn,
        "graph_parent_expansion",
        lambda: graph.case_ids_to_hits(conn, ids, filters=filters, config=config),
        diagnostics,
    )

    return [
        replace(
            hit,
            score=float(hop_by_case[hit.case_id]),
            meta={
                **hit.meta,
                "hop": hop_by_case[hit.case_id],
                "retrieval_arm": "graph",
            },
        )
        for hit in hits
    ]


def _with_graph_direction(hits: list[Hit], direction: str) -> list[Hit]:
    return [replace(hit, meta={**hit.meta, "graph_direction": direction}) for hit in hits]


def _apply_case_dedup(
    hits: list[Hit], query_class: QueryClass, config: RetrieverConfig
) -> list[Hit]:
    """Apply case-diversity post-filter when enabled."""
    if not config.case_diversity_enabled:
        return hits
    return case_dedup(hits, max_per_case=config.case_max_per_class(query_class.value))


def _run_pass2_rrf(
    seeds: list[Hit],
    graph_hits: list[Hit],
    *,
    limit: int,
    config: RetrieverConfig,
) -> list[Hit]:
    """Pass-2 RRF with typed kwargs unpacked from config.

    ``application_pass2_kwargs()`` returns ``dict[str, float | int]``,
    which cannot be safely spread with ``**`` into ``rrf_pass2`` —
    pyright loses the key-to-parameter mapping and reports
    ``reportArgumentType`` on every keyword (including ``config``,
    which is ``RetrieverConfig | None``, not a number). Unpacking
    field-by-field preserves type-checking at the call site.
    """
    kwargs = config.application_pass2_kwargs()
    return rrf_pass2(
        seeds,
        graph_hits,
        weight_seeds=float(kwargs["weight_seeds"]),
        weight_graph=float(kwargs["weight_graph"]),
        k_seeds=int(kwargs["k_seeds"]),
        k_graph=int(kwargs["k_graph"]),
        limit=limit,
    )


# ── HOLDING ────────────────────────────────────────────────────────────


def _route_holding(
    conn: Connection | None,
    embedder: Any,
    query: str,
    filters: Filters,
    top_k: int,
    started: float,
    config: RetrieverConfig,
) -> RouterResult:
    """HOLDING: dense + sparse + optional exact (if citation present)."""
    diagnostics: list[ArmDiagnostic] = []
    arms_used: list[str] = []

    dense_hits: list[Hit] = []
    try:
        query_vec = _embed_query(embedder, query)
        arms_used.append("dense")
        dense_hits = _run_and_record(
            conn,
            "dense",
            lambda: dense.search(
                conn,
                query_vec,
                filters=filters,
                limit=config.dense_limit,
                config=config,
            ),
            diagnostics,
        )
    except Exception:
        logger.exception("HOLDING dense arm unavailable; retaining lexical retrieval")

    arms_used.append("sparse")
    sparse_hits = _run_and_record(
        conn,
        "sparse",
        lambda: sparse.search(conn, query, filters=filters, limit=config.sparse_limit),
        diagnostics,
    )

    citation = extract_citation(query)
    exact_hits: list[Hit] = []
    if citation:
        arms_used.append("exact")
        exact_hits = _run_and_record(
            conn,
            "exact",
            lambda: exact.search(
                conn,
                citation,
                limit=20,
                filters=filters,
                config=config,
            ),
            diagnostics,
        )

    fused = rrf_pass1(
        {
            "dense": dense_hits,
            "sparse": sparse_hits,
            "exact": exact_hits,
        },
        config.holding_rrf_k(),
        limit=top_k * config.rrf_output_multiplier,
    )
    fused = _apply_case_dedup(fused, QueryClass.HOLDING, config)

    logger.info(
        "HOLDING dense=%d sparse=%d exact=%d pre_dedup=%d",
        len(dense_hits),
        len(sparse_hits),
        len(exact_hits),
        len(fused),
    )

    return RouterResult(
        query=query,
        query_class=QueryClass.HOLDING,
        arms_used=arms_used,
        hits=fused[:top_k],
        latency_ms=(time.monotonic() - started) * 1000,
        arm_diagnostics=diagnostics,
    )


# ── LOOKUP ─────────────────────────────────────────────────────────────


def _route_lookup(
    conn: Connection | None,
    query: str,
    filters: Filters,
    top_k: int,
    started: float,
    config: RetrieverConfig,
) -> RouterResult:
    """LOOKUP: sparse + exact/resolution + incoming graph, one-pass RRF.

    When the target case cannot be resolved, graph traversal is
    skipped. Sparse neighbors are not proof of legal identity — using
    them as graph seeds produces unrelated neighborhoods that look
    like evidence but are not.
    """
    diagnostics: list[ArmDiagnostic] = []
    arms_used: list[str] = ["sparse"]

    sparse_hits = _run_and_record(
        conn,
        "sparse",
        lambda: sparse.search(conn, query, filters=filters, limit=config.sparse_limit),
        diagnostics,
    )

    citation = extract_citation(query)
    exact_hits: list[Hit] = []
    if citation:
        arms_used.append("exact")
        exact_hits = _run_and_record(
            conn,
            "exact",
            lambda: exact.search(
                conn,
                citation,
                limit=20,
                filters=filters,
                config=config,
            ),
            diagnostics,
        )

    resolved_ids = _resolve_target_ids(conn, query, exact_hits, diagnostics, config)

    resolved_hits: list[Hit] = []
    if not exact_hits and resolved_ids:
        resolved_hits = _resolution_hits(conn, resolved_ids, filters, diagnostics, config)
        if resolved_hits:
            arms_used.append("case_resolution")

    graph_hits: list[Hit] = []
    if resolved_ids:
        arms_used.append("graph_1hop")
        graph_results = _run_and_record(
            conn,
            "graph_1hop",
            lambda: graph.search(
                conn,
                resolved_ids,
                direction="incoming",
                depth=1,
                cap_per_hop=config.lookup_graph_cap_per_hop,
                limit=config.graph_limit,
                filters=filters,
                config=config,
            ),
            diagnostics,
        )
        graph_hits = _graph_hits(conn, graph_results, filters, diagnostics, config)
        graph_hits = _with_graph_direction(graph_hits, "incoming")
    else:
        logger.warning(
            "LOOKUP graph skipped: target case could not be resolved "
            "(sparse neighbors are not legal identity)"
        )

    exact_arm_hits = exact_hits + resolved_hits

    fused = rrf_pass1(
        {
            "sparse": sparse_hits,
            "exact": exact_arm_hits,
            "graph_1hop": graph_hits,
        },
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


# ── VALIDITY ───────────────────────────────────────────────────────────


def _route_validity(
    conn: Connection | None,
    embedder: Any,
    query: str,
    filters: Filters,
    top_k: int,
    started: float,
    config: RetrieverConfig,
) -> RouterResult:
    """VALIDITY: resolve target -> incoming graph -> dense over citers.

    When the target case cannot be resolved, graph traversal is
    skipped. CITES edges identify citers but do not establish that
    the target was overruled — validity verification is a downstream
    stage. But the retrieval layer must still start from the actual
    target case, not from arbitrary sparse neighbors.
    """
    diagnostics: list[ArmDiagnostic] = []
    arms_used: list[str] = ["sparse"]

    sparse_hits = _run_and_record(
        conn,
        "sparse",
        lambda: sparse.search(conn, query, filters=filters, limit=config.sparse_limit),
        diagnostics,
    )

    citation = extract_citation(query)
    exact_hits: list[Hit] = []
    if citation:
        arms_used.append("exact")
        exact_hits = _run_and_record(
            conn,
            "exact",
            lambda: exact.search(
                conn,
                citation,
                limit=20,
                filters=filters,
                config=config,
            ),
            diagnostics,
        )

    resolved_ids = _resolve_target_ids(conn, query, exact_hits, diagnostics, config)

    if not resolved_ids:
        logger.warning(
            "VALIDITY graph skipped: target case could not be resolved "
            "(sparse neighbors are not legal identity)"
        )
        fused = rrf_pass1(
            {"sparse": sparse_hits, "exact": exact_hits},
            {
                "sparse": config.sparse_k_validity,
                "exact": config.exact_k,
            },
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
    graph_results = _run_and_record(
        conn,
        "graph_1hop",
        lambda: graph.search(
            conn,
            resolved_ids,
            direction="incoming",
            depth=1,
            cap_per_hop=config.validity_graph_cap_per_hop,
            limit=config.graph_limit,
            filters=filters,
            config=config,
        ),
        diagnostics,
    )
    graph_hits = _graph_hits(conn, graph_results, filters, diagnostics, config)
    graph_hits = _with_graph_direction(graph_hits, "incoming")

    dense_citer_hits: list[Hit] = []

    graph_case_ids: list[int] = []
    seen_graph_ids: set[int] = set()
    for row in graph_results:
        case_id = row.get("case_id")
        if isinstance(case_id, bool) or not isinstance(case_id, int):
            raise RuntimeError(f"invalid graph case ID: {case_id!r}")
        if case_id not in seen_graph_ids:
            graph_case_ids.append(case_id)
            seen_graph_ids.add(case_id)
        if len(graph_case_ids) >= config.graph_limit:
            break

    if graph_case_ids:
        citer_filters = filters.with_case_ids(graph_case_ids)
        if citer_filters.matches_nothing:
            logger.info(
                "VALIDITY dense_citers skipped: graph candidates "
                "do not intersect the requested case-ID filter"
            )
        else:
            arms_used.append("dense_citers")
            try:
                query_vec = _embed_query(embedder, query)
                dense_citer_hits = _run_and_record(
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
    else:
        logger.info("VALIDITY dense_citers skipped: graph returned no citing cases")

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


# ── APPLICATION ────────────────────────────────────────────────────────


def _route_application(
    conn: Connection | None,
    embedder: Any,
    query: str,
    filters: Filters,
    top_k: int,
    started: float,
    config: RetrieverConfig,
) -> RouterResult:
    """APPLICATION: dense + sparse -> resolve target -> incoming graph -> pass 2.

    Two sub-cases:

    * **Case-ref APPLICATION** (query names a case): the resolved
      case IDs seed graph traversal with
      ``config.application_graph_direction`` (default incoming). The
      target case is filtered out of pass-1 so that results are
      *applications* of X, not X itself. This is the audit fix —
      pre-hardening code seeded from pass-1 and used outgoing
      direction, returning cases X cites instead of cases citing X.

    * **Doctrine APPLICATION** (no case name): no seed available —
      routes to dense + sparse only. Sub-cases logged separately for
      no-name vs unresolved-name vs zero-citers, per the module
      docstring.
    """
    diagnostics: list[ArmDiagnostic] = []
    arms_used: list[str] = ["dense", "sparse"]

    dense_hits: list[Hit] = []
    try:
        query_vec = _embed_query(embedder, query)
        dense_hits = _run_and_record(
            conn,
            "dense",
            lambda: dense.search(
                conn,
                query_vec,
                filters=filters,
                limit=config.dense_limit,
                config=config,
            ),
            diagnostics,
        )
    except Exception:
        logger.exception("APPLICATION dense arm unavailable; using sparse results")

    sparse_hits = _run_and_record(
        conn,
        "sparse",
        lambda: sparse.search(conn, query, filters=filters, limit=config.sparse_limit),
        diagnostics,
    )

    pass1 = rrf_pass1(
        {"dense": dense_hits, "sparse": sparse_hits},
        config.application_pass1_rrf_k(),
        limit=top_k * config.rrf_output_multiplier,
    )

    # Citation hits are used for seed resolution only — they do not
    # contribute to fusion. The "exact" diagnostic entry appears in
    # arm_diagnostics but not in arms_used, matching the attempted vs
    # contributed distinction documented in the module docstring.
    citation = extract_citation(query)
    citation_hits: list[Hit] = []
    if citation:
        citation_hits = _run_and_record(
            conn,
            "exact",
            lambda: exact.search(
                conn,
                citation,
                limit=20,
                filters=filters,
                config=config,
            ),
            diagnostics,
        )

    resolved_ids = _resolve_target_ids(conn, query, citation_hits, diagnostics, config)

    # ── Fallback: no seed available (doctrine or unresolved case) ──
    if not resolved_ids:
        case_ref = extract_case_reference(query)
        if case_ref:
            logger.warning(
                "APPLICATION graph skipped: case reference %r not resolvable in corpus",
                case_ref,
            )
        else:
            logger.info("APPLICATION without case reference: doctrine-only, graph skipped")
        fused = _apply_case_dedup(pass1, QueryClass.APPLICATION, config)
        return RouterResult(
            query=query,
            query_class=QueryClass.APPLICATION,
            arms_used=arms_used,
            hits=fused[:top_k],
            latency_ms=(time.monotonic() - started) * 1000,
            arm_diagnostics=diagnostics,
        )

    # Case-ref APPLICATION: filter target case from pass-1 so results
    # are applications of X, not X itself. Graph traversal already
    # excludes seeds server-side (graph._one_hop skips seed IDs).
    resolved_set = set(resolved_ids)
    pass1 = [h for h in pass1 if h.case_id not in resolved_set]

    graph_arm_name = "graph_2hop" if config.application_2hop_enabled else "graph_1hop"
    depth = config.application_graph_depth

    logger.info(
        "APPLICATION using %s, direction=%s, depth=%d, seeds=%d",
        graph_arm_name,
        config.application_graph_direction,
        depth,
        len(resolved_ids),
    )

    arms_used.append(graph_arm_name)
    graph_results = _run_and_record(
        conn,
        graph_arm_name,
        lambda: graph.search(
            conn,
            resolved_ids,
            direction=config.application_graph_direction,
            depth=depth,
            cap_per_hop=config.application_graph_cap_per_hop,
            limit=config.application_graph_limit,
            filters=filters,
            config=config,
        ),
        diagnostics,
    )

    graph_hits = _graph_hits(conn, graph_results, filters, diagnostics, config)
    graph_hits = _with_graph_direction(graph_hits, config.application_graph_direction)

    if graph_hits:
        fused = _run_pass2_rrf(
            pass1,
            graph_hits,
            limit=top_k * config.rrf_output_multiplier,
            config=config,
        )
    else:
        logger.warning(
            "APPLICATION %s returned no citers of target; retaining filtered first-pass results",
            graph_arm_name,
        )
        fused = pass1

    fused = _apply_case_dedup(fused, QueryClass.APPLICATION, config)

    logger.info(
        "APPLICATION dense=%d sparse=%d %s=%d resolved_seeds=%d",
        len(dense_hits),
        len(sparse_hits),
        graph_arm_name,
        len(graph_hits),
        len(resolved_ids),
    )

    return RouterResult(
        query=query,
        query_class=QueryClass.APPLICATION,
        arms_used=arms_used,
        hits=fused[:top_k],
        latency_ms=(time.monotonic() - started) * 1000,
        arm_diagnostics=diagnostics,
    )


# ── Public dispatch ────────────────────────────────────────────────────


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

    ``config`` is optional; when None, ``default_config()`` (env-driven)
    is used. All per-arm limits, RRF k values, graph direction, case
    diversity caps, and the 2-hop gate come from this object.
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

    started = time.monotonic()
    query_class = classify(normalized_query)

    logger.info(
        "query_class=%s route starting (2hop_enabled=%s)",
        query_class.value,
        config.application_2hop_enabled,
    )

    if filters.matches_nothing:
        logger.info(
            "%s route skipped: explicit empty case-ID filter",
            query_class.value,
        )
        return RouterResult(
            query=normalized_query,
            query_class=query_class,
            arms_used=[],
            hits=[],
            latency_ms=(time.monotonic() - started) * 1000,
            arm_diagnostics=[],
        )

    if query_class == QueryClass.HOLDING:
        return _route_holding(conn, embedder, normalized_query, filters, top_k, started, config)

    if query_class == QueryClass.LOOKUP:
        return _route_lookup(conn, normalized_query, filters, top_k, started, config)

    if query_class == QueryClass.VALIDITY:
        return _route_validity(conn, embedder, normalized_query, filters, top_k, started, config)

    if query_class == QueryClass.APPLICATION:
        return _route_application(conn, embedder, normalized_query, filters, top_k, started, config)

    raise RuntimeError(f"unhandled query class: {query_class!r}")


__all__ = [
    "ArmDiagnostic",
    "RouterResult",
    "route",
]
