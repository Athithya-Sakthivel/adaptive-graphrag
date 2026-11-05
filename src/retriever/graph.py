"""Citation-graph retrieval using PostgreSQL recursive CTEs.

Filters are applied inside every graph SQL query, before LIMIT. This
avoids the filter-after-limit recall loss that silently discards
eligible candidates.

Two parameter names appear in graph SQL:
    %(seeds)s / %(case_ids)s   — the IDs to traverse from / expand
    %(filter_case_ids)s        — the user's original case-ID restriction

These are separate because graph traversal can return IDs that the
user's case-ID filter would exclude. The SQL applies
%(filter_case_ids)s at every hop so out-of-scope cases never enter
the traversal or the parent expansion.

Graph ranking
-------------
The ORDER BY of every case-ranking graph query is driven by
``config.graph_rank_by``:

    "in_graph_degree"
        ORDER BY COUNT(case_edges WHERE source_id = X) DESC.
        Cheap, in-graph, reflects how often the case is cited
        *within the sampled corpus*. Default.

    "citation_count"
        ORDER BY cases.citation_count DESC NULLS LAST.
        CourtListener-wide count. Use with care: a case with 5000
        global citations but 0 in-graph citers ranks above a real
        hub.

    "case_id"
        ORDER BY case_id ASC. Deterministic, semantically arbitrary.
        Ablation only.

``CASE_IDS_TO_PARENTS`` is deliberately excluded from this
rank-switch. Its ORDER BY under ``DISTINCT ON (c.case_id)`` controls
*which parent chunk is selected per case* (highest section_priority
first), not case ordering. Output order of ``case_ids_to_hits``
follows the caller's input ``case_ids`` order, which is already
ranked by ``search_1hop`` / ``search_2hop``.

Fan-out caps
------------
Both 1-hop and 2-hop queries apply a per-seed cap before the LIMIT.
When a query returns exactly ``limit`` rows, ``logger.info`` records
the event so callers can distinguish "no results" from "cap hit."
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal

from psycopg import Connection
from psycopg.rows import dict_row

from .filters import Filters
from .fusion import Hit

if TYPE_CHECKING:
    from .config import RetrieverConfig


logger = logging.getLogger(__name__)

Direction = Literal["incoming", "outgoing", "both"]
GraphRankBy = Literal["in_graph_degree", "citation_count", "case_id"]


# ── Rank clause builders ───────────────────────────────────────────────
#
# `case_id_expr` is ALWAYS an internally-controlled column reference
# (e.g. "e.source_id", "candidate.case_id", "walk.case_id"). It is
# never user input. `rank_by` is validated against a closed set.


def _rank_clause(rank_by: str, case_id_expr: str) -> str:
    """ORDER BY clause for a case-level SQL query."""
    if rank_by == "case_id":
        return f"{case_id_expr} ASC"

    if rank_by == "in_graph_degree":
        return (
            f"(SELECT COUNT(*) FROM public.case_edges AS ce_rank "
            f"WHERE ce_rank.source_id = {case_id_expr}) DESC, "
            f"{case_id_expr} ASC"
        )

    if rank_by == "citation_count":
        return (
            f"(SELECT ca_rank.citation_count FROM public.cases AS ca_rank "
            f"WHERE ca_rank.case_id = {case_id_expr}) DESC NULLS LAST, "
            f"{case_id_expr} ASC"
        )

    raise ValueError(f"invalid graph_rank_by: {rank_by!r}")


def _rank_signal_expr(rank_by: str, case_id_expr: str) -> str:
    """Scalar rank-signal column for the 2-hop merged result.

    Returned as a SELECT column so the per-seed Python merge can
    re-sort with the same semantics as the SQL ORDER BY.
    """
    if rank_by == "case_id":
        return "0::bigint"

    if rank_by == "in_graph_degree":
        return (
            f"(SELECT COUNT(*) FROM public.case_edges AS ce_rs "
            f"WHERE ce_rs.source_id = {case_id_expr})"
        )

    if rank_by == "citation_count":
        return (
            f"(SELECT ca_rs.citation_count FROM public.cases AS ca_rs "
            f"WHERE ca_rs.case_id = {case_id_expr})"
        )

    raise ValueError(f"invalid graph_rank_by: {rank_by!r}")


# ── 1-hop SQL builders ─────────────────────────────────────────────────


def _build_1hop_directional_sql(direction: str, rank_by: str) -> str:
    if direction == "incoming":
        edge_col = "source_id"
        match_col = "target_id"
    elif direction == "outgoing":
        edge_col = "target_id"
        match_col = "source_id"
    else:
        raise ValueError(f"invalid directional 1-hop: {direction!r}")

    rank_clause = _rank_clause(rank_by, f"e.{edge_col}")

    return f"""
SELECT DISTINCT e.{edge_col} AS case_id
FROM public.case_edges AS e
JOIN public.cases AS ca ON ca.case_id = e.{edge_col}
WHERE e.{match_col} = ANY(%(seeds)s::bigint[])
  AND e.relation = 'CITES'
  AND (%(court)s::text IS NULL OR ca.court = %(court)s)
  AND (%(jurisdiction)s::text IS NULL OR ca.jurisdiction = %(jurisdiction)s)
  AND (%(published_only)s = FALSE OR lower(btrim(ca.precedential_status)) = 'published')
  AND (%(date_from)s::date IS NULL OR ca.date >= %(date_from)s)
  AND (%(date_to)s::date IS NULL OR ca.date <= %(date_to)s)
  AND (%(filter_case_ids)s::bigint[] IS NULL OR e.{edge_col} = ANY(%(filter_case_ids)s::bigint[]))
ORDER BY {rank_clause}
LIMIT %(limit)s
"""


def _build_1hop_both_sql(rank_by: str) -> str:
    rank_clause = _rank_clause(rank_by, "u.case_id")

    return f"""
SELECT u.case_id
FROM (
    SELECT e.source_id AS case_id
    FROM public.case_edges AS e
    JOIN public.cases AS ca ON ca.case_id = e.source_id
    WHERE e.target_id = ANY(%(seeds)s::bigint[])
      AND e.relation = 'CITES'
      AND (%(court)s::text IS NULL OR ca.court = %(court)s)
      AND (%(jurisdiction)s::text IS NULL OR ca.jurisdiction = %(jurisdiction)s)
      AND (%(published_only)s = FALSE OR lower(btrim(ca.precedential_status)) = 'published')
      AND (%(date_from)s::date IS NULL OR ca.date >= %(date_from)s)
      AND (%(date_to)s::date IS NULL OR ca.date <= %(date_to)s)
      AND (%(filter_case_ids)s::bigint[] IS NULL OR e.source_id = ANY(%(filter_case_ids)s::bigint[]))

    UNION

    SELECT e.target_id AS case_id
    FROM public.case_edges AS e
    JOIN public.cases AS ca ON ca.case_id = e.target_id
    WHERE e.source_id = ANY(%(seeds)s::bigint[])
      AND e.relation = 'CITES'
      AND (%(court)s::text IS NULL OR ca.court = %(court)s)
      AND (%(jurisdiction)s::text IS NULL OR ca.jurisdiction = %(jurisdiction)s)
      AND (%(published_only)s = FALSE OR lower(btrim(ca.precedential_status)) = 'published')
      AND (%(date_from)s::date IS NULL OR ca.date >= %(date_from)s)
      AND (%(date_to)s::date IS NULL OR ca.date <= %(date_to)s)
      AND (%(filter_case_ids)s::bigint[] IS NULL OR e.target_id = ANY(%(filter_case_ids)s::bigint[]))
) AS u
ORDER BY {rank_clause}
LIMIT %(limit)s
"""


def _build_1hop_sql(direction: Direction, rank_by: str) -> str:
    if direction == "both":
        return _build_1hop_both_sql(rank_by)
    if direction in ("incoming", "outgoing"):
        return _build_1hop_directional_sql(direction, rank_by)
    raise ValueError(f"invalid 1-hop direction: {direction!r}")


# ── Parent expansion ──────────────────────────────────────────────────


CASE_IDS_TO_PARENTS = """
SELECT DISTINCT ON (c.case_id)
    c.chunk_id,
    c.case_id,
    c.court,
    c.date,
    c.citation,
    LEFT(c.text_original, 200) AS preview
FROM public.chunks AS c
JOIN public.cases AS ca ON ca.case_id = c.case_id
WHERE c.case_id = ANY(%(case_ids)s::bigint[])
  AND c.is_parent = TRUE
  AND (%(court)s::text IS NULL OR ca.court = %(court)s)
  AND (%(jurisdiction)s::text IS NULL OR ca.jurisdiction = %(jurisdiction)s)
  AND (%(published_only)s = FALSE OR lower(btrim(ca.precedential_status)) = 'published')
  AND (%(date_from)s::date IS NULL OR ca.date >= %(date_from)s)
  AND (%(date_to)s::date IS NULL OR ca.date <= %(date_to)s)
  AND (%(filter_case_ids)s::bigint[] IS NULL OR c.case_id = ANY(%(filter_case_ids)s::bigint[]))
ORDER BY
    c.case_id ASC,
    c.section_priority DESC,
    c.section_index ASC,
    c.chunk_id ASC
"""


# ── Validation ─────────────────────────────────────────────────────────


def _validate_ids(case_ids: list[int]) -> list[int]:
    if not isinstance(case_ids, list):
        raise TypeError("seed_case_ids must be a list")

    normalized: list[int] = []
    for case_id in case_ids:
        if isinstance(case_id, bool) or not isinstance(case_id, int):
            raise TypeError(f"case ID must be an integer: {case_id!r}")
        if case_id <= 0:
            raise ValueError(f"case ID must be positive: {case_id}")
        normalized.append(case_id)

    return list(dict.fromkeys(normalized))


def _validate_positive(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _date_string(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return str(value.isoformat())
    return str(value)


def _resolve_rank_by(config: RetrieverConfig | None) -> str:
    if config is None:
        from .config import default_config

        config = default_config()
    return config.graph_rank_by


def _graph_params(
    filters: Filters,
    *,
    seeds: list[int] | None = None,
    expand_ids: list[int] | None = None,
    limit: int | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Build SQL parameters for a graph query.

    Splits the user's case_ids restriction (``filter_case_ids`` in
    SQL) from the traversal targets (``seeds`` or ``case_ids`` in
    SQL).
    """
    params = filters.sql_params()
    user_case_filter = params.pop("case_ids")

    if seeds is not None:
        params["seeds"] = seeds
    if expand_ids is not None:
        params["case_ids"] = expand_ids
    if limit is not None:
        params["limit"] = limit

    params["filter_case_ids"] = user_case_filter
    params.update(extra)
    return params


# ── SQL cache (per direction × rank_by) ────────────────────────────────


_1HOP_SQL_CACHE: dict[tuple[str, str], str] = {}
_2HOP_SQL_CACHE: dict[tuple[str, str], str] = {}


def _get_1hop_sql(direction: Direction, rank_by: str) -> str:
    key = (direction, rank_by)
    sql = _1HOP_SQL_CACHE.get(key)
    if sql is None:
        sql = _build_1hop_sql(direction, rank_by)
        _1HOP_SQL_CACHE[key] = sql
    return sql


def _get_2hop_sql(direction: Direction, rank_by: str) -> str:
    key = (direction, rank_by)
    sql = _2HOP_SQL_CACHE.get(key)
    if sql is None:
        sql = _build_2hop_sql(direction, rank_by)
        _2HOP_SQL_CACHE[key] = sql
    return sql


# ── 1-hop ──────────────────────────────────────────────────────────────


def _one_hop(
    conn: Connection,
    seed_case_ids: list[int],
    *,
    direction: Literal["incoming", "outgoing"],
    limit: int,
    filters: Filters,
    rank_by: str,
) -> list[dict[str, Any]]:
    if not seed_case_ids or filters.matches_nothing:
        return []

    sql = _get_1hop_sql(direction, rank_by)
    params = _graph_params(filters, seeds=seed_case_ids, limit=limit)

    with conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute(sql, params)
        rows = cursor.fetchall()

    if len(rows) >= limit:
        logger.info(
            "graph_1hop fan-out cap hit: direction=%s seeds=%d rows=%d limit=%d",
            direction,
            len(seed_case_ids),
            len(rows),
            limit,
        )

    seed_set = set(seed_case_ids)
    return [
        {"case_id": int(row["case_id"]), "hop": 1, "score": 0.0}
        for row in rows
        if int(row["case_id"]) not in seed_set
    ]


def search_1hop(
    conn: Connection,
    seed_case_ids: list[int],
    *,
    direction: Direction = "incoming",
    limit: int = 50,
    cap_per_hop: int = 50,
    filters: Filters | None = None,
    config: RetrieverConfig | None = None,
) -> list[dict[str, Any]]:
    """Find directly connected opinions.

    Filters are applied inside the SQL query, before LIMIT. Cases
    outside the user's ``case_ids`` filter are never returned.

    Results are ordered by ``config.graph_rank_by`` (default
    ``in_graph_degree``). A fan-out cap hit is logged at INFO level.
    """
    seeds = _validate_ids(seed_case_ids)
    _validate_positive(limit, "limit")
    _validate_positive(cap_per_hop, "cap_per_hop")

    if direction not in ("incoming", "outgoing", "both"):
        raise ValueError(f"invalid graph direction: {direction!r}")

    if filters is None:
        filters = Filters()
    if not isinstance(filters, Filters):
        raise TypeError("filters must be Filters or None")

    if not seeds or filters.matches_nothing:
        return []

    rank_by = _resolve_rank_by(config)
    effective_limit = min(limit, cap_per_hop * len(seeds))

    if direction in ("incoming", "outgoing"):
        return _one_hop(
            conn,
            seeds,
            direction=direction,
            limit=effective_limit,
            filters=filters,
            rank_by=rank_by,
        )

    # direction == "both": single combined query, ranked across both
    # directions. (The old two-query merge preserved incoming-first
    # order, which mis-ranked the combined set.)
    sql = _get_1hop_sql("both", rank_by)
    params = _graph_params(filters, seeds=seeds, limit=effective_limit)

    with conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute(sql, params)
        rows = cursor.fetchall()

    if len(rows) >= effective_limit:
        logger.info(
            "graph_1hop fan-out cap hit: direction=both seeds=%d rows=%d limit=%d",
            len(seeds),
            len(rows),
            effective_limit,
        )

    seed_set = set(seeds)
    return [
        {"case_id": int(row["case_id"]), "hop": 1, "score": 0.0}
        for row in rows
        if int(row["case_id"]) not in seed_set
    ]


# ── 2-hop ──────────────────────────────────────────────────────────────


def _neighbor_sql(
    direction: Literal["incoming", "outgoing", "both"],
    reference: str,
) -> str:
    """Neighbor subquery with all filter predicates included.

    ``reference`` is an internally controlled placeholder string
    (``'%(seed)s'`` or ``'walk.case_id'``); it is never user input.
    """

    def filters_clause(case_col: str) -> str:
        return f"""
        AND (%(court)s::text IS NULL OR ca.court = %(court)s)
        AND (%(jurisdiction)s::text IS NULL OR ca.jurisdiction = %(jurisdiction)s)
        AND (%(published_only)s = FALSE OR lower(btrim(ca.precedential_status)) = 'published')
        AND (%(date_from)s::date IS NULL OR ca.date >= %(date_from)s)
        AND (%(date_to)s::date IS NULL OR ca.date <= %(date_to)s)
        AND (%(filter_case_ids)s::bigint[] IS NULL OR e.{case_col} = ANY(%(filter_case_ids)s::bigint[]))
        """

    if direction == "outgoing":
        return f"""
            SELECT e.target_id AS case_id
            FROM public.case_edges AS e
            JOIN public.cases AS ca ON ca.case_id = e.target_id
            WHERE e.source_id = {reference}
              AND e.relation = 'CITES'
              {filters_clause("target_id")}
        """

    if direction == "incoming":
        return f"""
            SELECT e.source_id AS case_id
            FROM public.case_edges AS e
            JOIN public.cases AS ca ON ca.case_id = e.source_id
            WHERE e.target_id = {reference}
              AND e.relation = 'CITES'
              {filters_clause("source_id")}
        """

    if direction == "both":
        return f"""
            SELECT neighbors.case_id
            FROM (
                SELECT e.target_id AS case_id
                FROM public.case_edges AS e
                JOIN public.cases AS ca ON ca.case_id = e.target_id
                WHERE e.source_id = {reference}
                  AND e.relation = 'CITES'
                  {filters_clause("target_id")}

                UNION

                SELECT e.source_id AS case_id
                FROM public.case_edges AS e
                JOIN public.cases AS ca ON ca.case_id = e.source_id
                WHERE e.target_id = {reference}
                  AND e.relation = 'CITES'
                  {filters_clause("source_id")}
            ) AS neighbors
        """

    raise ValueError(f"invalid direction: {direction!r}")


def _build_2hop_sql(direction: Direction, rank_by: str) -> str:
    first_hop = _neighbor_sql(direction, "%(seed)s")
    next_hop = _neighbor_sql(direction, "walk.case_id")

    rank_clause_initial = _rank_clause(rank_by, "candidate.case_id")
    rank_clause_next = _rank_clause(rank_by, "candidate.case_id")
    rank_signal = _rank_signal_expr(rank_by, "walk.case_id")

    return f"""
    WITH RECURSIVE walk(case_id, hop) AS (
        SELECT
            initial_neighbors.case_id,
            1 AS hop
        FROM (
            SELECT candidate.case_id
            FROM ( {first_hop} ) AS candidate
            ORDER BY {rank_clause_initial}
            LIMIT %(cap_per_hop)s
        ) AS initial_neighbors

        UNION

        SELECT
            next_neighbors.case_id,
            walk.hop + 1
        FROM walk
        CROSS JOIN LATERAL (
            SELECT candidate.case_id
            FROM ( {next_hop} ) AS candidate
            ORDER BY {rank_clause_next}
            LIMIT %(cap_per_hop)s
        ) AS next_neighbors
        WHERE walk.hop < %(depth)s
    )

    SELECT
        walk.case_id AS case_id,
        MIN(walk.hop) AS hop,
        {rank_signal} AS rank_signal
    FROM walk
    WHERE walk.case_id <> %(seed)s
    GROUP BY walk.case_id
    ORDER BY MIN(walk.hop) ASC, rank_signal DESC NULLS LAST, walk.case_id ASC
    LIMIT %(limit)s
    """


def search_2hop(
    conn: Connection,
    seed_case_ids: list[int],
    *,
    direction: Direction = "outgoing",
    depth: int = 2,
    cap_per_hop: int = 30,
    limit: int = 200,
    filters: Filters | None = None,
    config: RetrieverConfig | None = None,
) -> list[dict[str, Any]]:
    """Bounded, per-seed 2-hop citation traversal.

    Each seed receives its own recursive query so the initial fan-out
    cap applies independently to each seed. Filters are applied at
    every hop: cases outside the user's ``case_ids`` filter never
    enter the traversal.

    Results from all seeds are merged. If multiple paths reach the
    same case, the shortest hop distance wins. Within the same hop,
    ``config.graph_rank_by`` decides ordering (via ``rank_signal``
    returned from SQL and re-sorted in Python).
    """
    seeds = _validate_ids(seed_case_ids)

    _validate_positive(depth, "depth")
    _validate_positive(cap_per_hop, "cap_per_hop")
    _validate_positive(limit, "limit")

    if depth not in (1, 2):
        raise ValueError(f"depth must be 1 or 2, got {depth}")
    if direction not in ("incoming", "outgoing", "both"):
        raise ValueError(f"invalid graph direction: {direction!r}")

    if filters is None:
        filters = Filters()
    if not isinstance(filters, Filters):
        raise TypeError("filters must be Filters or None")

    if not seeds or filters.matches_nothing:
        return []

    rank_by = _resolve_rank_by(config)
    sql = _get_2hop_sql(direction, rank_by)

    # case_id -> (hop, rank_signal)
    best: dict[int, tuple[int, int]] = {}

    for seed in seeds:
        params = _graph_params(
            filters,
            limit=limit,
            seed=seed,
            depth=depth,
            cap_per_hop=cap_per_hop,
        )
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(sql, params)
            rows = cursor.fetchall()

        if len(rows) >= limit:
            logger.info(
                "graph_2hop fan-out cap hit: direction=%s seed=%d rows=%d limit=%d",
                direction,
                seed,
                len(rows),
                limit,
            )

        for row in rows:
            case_id = int(row["case_id"])
            hop = int(row["hop"])

            if case_id in seeds:
                continue

            raw_signal = row.get("rank_signal")
            rank_signal = int(raw_signal) if raw_signal is not None else 0

            if case_id not in best or hop < best[case_id][0]:
                best[case_id] = (hop, rank_signal)

    # Merge across seeds: shortest hop first; within a hop, highest
    # rank_signal first (via negation); case_id as deterministic
    # tiebreak.
    ordered = sorted(
        best.items(),
        key=lambda item: (item[1][0], -item[1][1], item[0]),
    )[:limit]

    return [{"case_id": case_id, "hop": hop, "score": 0.0} for case_id, (hop, _signal) in ordered]


# ── Parent expansion ───────────────────────────────────────────────────


def case_ids_to_hits(
    conn: Connection,
    case_ids: list[int],
    scores: Mapping[int, float] | None = None,
    *,
    filters: Filters | None = None,
    config: RetrieverConfig | None = None,
) -> list[Hit]:
    """Expand case IDs into one parent Hit per case.

    The ``filters`` parameter applies the user's metadata restrictions
    (court, jurisdiction, published_only, dates, case_ids) during
    expansion. Cases that were returned by graph traversal but fail
    the user's filters are excluded from the output.

    Parent selection (per case, fixed regardless of config):
        Highest ``section_priority``, then lowest ``section_index``,
        then lowest ``chunk_id``. This is a chunk-selection rule, not
        a case-ranking rule — deliberately not driven by
        ``config.graph_rank_by``.

    Return order follows input ``case_ids`` order, which is already
    ranked by the caller's ``search_1hop`` / ``search_2hop`` call.

    ``config`` is accepted for API uniformity with the other graph
    entry points; it is unused here because neither chunk selection
    nor output order depends on it.
    """
    ids = _validate_ids(case_ids)

    if not ids:
        return []

    if filters is None:
        filters = Filters()
    if not isinstance(filters, Filters):
        raise TypeError("filters must be Filters or None")

    if filters.matches_nothing:
        return []

    score_map = dict(scores) if scores is not None else {}
    for case_id, score in score_map.items():
        if isinstance(case_id, bool) or not isinstance(case_id, int):
            raise TypeError("score-map keys must be integer case IDs")
        if not isinstance(score, (int, float)):
            raise TypeError("case scores must be numeric")

    params = _graph_params(filters, expand_ids=ids)

    with conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute(CASE_IDS_TO_PARENTS, params)
        rows = cursor.fetchall()

    hits = [
        Hit(
            chunk_id=str(row["chunk_id"]),
            case_id=int(row["case_id"]),
            is_parent=True,
            score=float(score_map.get(int(row["case_id"]), 0.0)),
            rank=0,
            meta={
                "preview": row["preview"] or "",
                "court": row["court"],
                "date": _date_string(row["date"]),
                "citation": row["citation"],
            },
        )
        for row in rows
    ]

    order = {case_id: i for i, case_id in enumerate(ids)}
    hits.sort(key=lambda hit: order.get(hit.case_id, len(ids)))
    return hits


# ── Public dispatcher ──────────────────────────────────────────────────


def search(
    conn: Connection,
    seed_case_ids: list[int],
    *,
    direction: Direction = "incoming",
    depth: Literal[1, 2] = 1,
    cap_per_hop: int = 50,
    limit: int = 200,
    filters: Filters | None = None,
    config: RetrieverConfig | None = None,
) -> list[dict[str, Any]]:
    """Public graph-search dispatcher.

    Routes to ``search_1hop`` or ``search_2hop`` based on ``depth``.
    Both delegate ranking to ``config.graph_rank_by`` (resolved by
    each function; ``config=None`` falls through to the module-level
    ``default_config()``).
    """

    if depth == 1:
        return search_1hop(
            conn,
            seed_case_ids,
            direction=direction,
            limit=limit,
            cap_per_hop=cap_per_hop,
            filters=filters,
            config=config,
        )

    if depth == 2:
        return search_2hop(
            conn,
            seed_case_ids,
            direction=direction,
            depth=2,
            cap_per_hop=cap_per_hop,
            limit=limit,
            filters=filters,
            config=config,
        )

    raise ValueError(f"depth must be 1 or 2, got {depth!r}")


__all__ = [
    "Direction",
    "GraphRankBy",
    "CASE_IDS_TO_PARENTS",
    "search_1hop",
    "search_2hop",
    "case_ids_to_hits",
    "search",
]
