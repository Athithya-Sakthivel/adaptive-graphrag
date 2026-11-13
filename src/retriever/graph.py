"""Citation-graph retrieval using PostgreSQL recursive CTEs.

Filters are applied inside every SQL query (before LIMIT) to avoid
the filter-after-limit recall loss that silently discards eligible
candidates when HNSW-style pruning applies.

Ranking is driven by config.graph_rank_by:
    in_graph_degree — COUNT(case_edges WHERE source_id = X)
                      Cheap, in-graph, reflects local hub status.
    citation_count — CourtListener's global count. Use with care:
                     a case with 5000 global citations but 0 in-graph
                     citers ranks above a real hub.
    case_id        — numeric ordering. Deterministic, arbitrary.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

from psycopg import Connection
from psycopg.rows import dict_row

from .filters import Filters

if TYPE_CHECKING:
    from .config import RetrieverConfig

logger = logging.getLogger("retriever.graph")

Direction = Literal["incoming", "outgoing", "both"]


def _rank_clause(rank_by: str, case_id_expr: str) -> str:
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


def _filters_clause(edge_col: str) -> str:
    return f"""
      AND (%(court)s::text IS NULL OR ca.court = %(court)s)
      AND (%(jurisdiction)s::text IS NULL OR ca.jurisdiction = %(jurisdiction)s)
      AND (%(published_only)s = FALSE OR lower(btrim(ca.precedential_status)) = 'published')
      AND (%(date_from)s::date IS NULL OR ca.date >= %(date_from)s)
      AND (%(date_to)s::date IS NULL OR ca.date <= %(date_to)s)
      AND (%(filter_case_ids)s::bigint[] IS NULL OR e.{edge_col} = ANY(%(filter_case_ids)s::bigint[]))
    """


def _build_1hop_sql(direction: Direction, rank_by: str) -> str:
    rank = _rank_clause(rank_by, "e.source_id" if direction == "incoming" else "e.target_id")
    edge_col = "source_id" if direction == "incoming" else "target_id"
    match_col = "target_id" if direction == "incoming" else "source_id"
    filters = _filters_clause(edge_col)

    if direction in ("incoming", "outgoing"):
        return f"""
            SELECT e.{edge_col} AS case_id, 1 AS hop
            FROM public.case_edges AS e
            JOIN public.cases AS ca ON ca.case_id = e.{edge_col}
            WHERE e.{match_col} = ANY(%(seeds)s::bigint[])
              AND e.relation = 'CITES'
              {filters}
            GROUP BY e.{edge_col}
            ORDER BY {rank}
            LIMIT %(limit)s
        """

    filters_in = _filters_clause("source_id")
    filters_out = _filters_clause("target_id")
    rank_u = _rank_clause(rank_by, "u.case_id")
    return f"""
        SELECT u.case_id, 1 AS hop
        FROM (
            SELECT e.source_id AS case_id
            FROM public.case_edges AS e
            JOIN public.cases AS ca ON ca.case_id = e.source_id
            WHERE e.target_id = ANY(%(seeds)s::bigint[])
              AND e.relation = 'CITES'
              {filters_in}
            UNION
            SELECT e.target_id AS case_id
            FROM public.case_edges AS e
            JOIN public.cases AS ca ON ca.case_id = e.target_id
            WHERE e.source_id = ANY(%(seeds)s::bigint[])
              AND e.relation = 'CITES'
              {filters_out}
        ) AS u
        ORDER BY {rank_u}
        LIMIT %(limit)s
    """


_1HOP_SQL_CACHE: dict[tuple[str, str], str] = {}


def _get_1hop_sql(direction: Direction, rank_by: str) -> str:
    key = (direction, rank_by)
    sql = _1HOP_SQL_CACHE.get(key)
    if sql is None:
        sql = _build_1hop_sql(direction, rank_by)
        _1HOP_SQL_CACHE[key] = sql
    return sql


_GRAPH_PARAMS_CACHE: dict = {}


def _graph_params(filters: Filters, seeds: list[int], limit: int) -> dict:
    params = filters.sql_params()
    user_case_filter = params.pop("case_ids")
    params["seeds"] = seeds
    params["limit"] = limit
    params["filter_case_ids"] = user_case_filter
    return params


def search_1hop(
    conn: Connection,
    seed_case_ids: list[int],
    *,
    direction: Direction = "incoming",
    limit: int = 50,
    cap_per_hop: int = 50,
    filters: Filters | None = None,
    config: RetrieverConfig | None = None,
) -> list[dict]:
    """Find directly connected opinions.

    Returns list of {"case_id": int, "hop": int} dicts, ordered by
    config.graph_rank_by. Filters are applied inside SQL before LIMIT.
    """
    if direction not in ("incoming", "outgoing", "both"):
        raise ValueError(f"invalid graph direction: {direction!r}")
    if limit < 1 or cap_per_hop < 1:
        raise ValueError("limit and cap_per_hop must be positive")

    seeds = list(
        dict.fromkeys(int(cid) for cid in seed_case_ids if isinstance(cid, int) and cid > 0)
    )
    if not seeds:
        return []

    if filters is None:
        filters = Filters()
    if not isinstance(filters, Filters):
        raise TypeError("filters must be Filters or None")
    if filters.matches_nothing:
        return []

    if config is None:
        from .config import default_config

        config = default_config()

    effective_limit = min(limit, cap_per_hop * len(seeds))
    sql = _get_1hop_sql(direction, config.graph_rank_by)
    params = _graph_params(filters, seeds, effective_limit)

    with conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute(sql, params)
        rows = cursor.fetchall()

    if len(rows) >= effective_limit:
        logger.info(
            "graph_1hop fan-out cap hit: direction=%s seeds=%d rows=%d limit=%d",
            direction,
            len(seeds),
            len(rows),
            effective_limit,
        )

    seed_set = set(seeds)
    return [
        {"case_id": int(row["case_id"]), "hop": 1}
        for row in rows
        if int(row["case_id"]) not in seed_set
    ]


def search(
    conn: Connection,
    seed_case_ids: list[int],
    *,
    direction: Direction = "incoming",
    depth: Literal[1] = 1,
    cap_per_hop: int = 50,
    limit: int = 200,
    filters: Filters | None = None,
    config: RetrieverConfig | None = None,
) -> list[dict]:
    """Public graph-search dispatcher. Currently 1-hop only."""
    if depth != 1:
        raise ValueError(f"depth must be 1, got {depth!r}")
    return search_1hop(
        conn,
        seed_case_ids,
        direction=direction,
        limit=limit,
        cap_per_hop=cap_per_hop,
        filters=filters,
        config=config,
    )


__all__ = [
    "Direction",
    "search_1hop",
    "search",
]
