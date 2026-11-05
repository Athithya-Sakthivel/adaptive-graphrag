"""Dense retrieval using PostgreSQL pgvector HNSW and halfvec(1024).

Uses `relaxed_order` iterative scanning (pgvector >= 0.8) with a
materialized-CTE re-sort to preserve distance ordering. Benchmark
evidence for this configuration:

    strict_order   loses at every selectivity; up to 100k rescored rows,
                   352 ms at 1 client / 742 ms at 8 clients
    relaxed_order  0.76-0.92 recall at 1% filter selectivity depending
                   on max_scan_tuples
    no iterative   ~0.012 recall at 1% selectivity (post-filter)
    seqscan        1.0 recall, 16-19 ms at small scale

A plain SELECT with ORDER BY + LIMIT would let the planner choose
seqscan at low selectivity, which has perfect recall but bypasses the
HNSW candidate graph entirely. `relaxed_order` + re-sort preserves
HNSW's breadth while guaranteeing deterministic ordering.

At 1% selectivity, the mitigation ladder is:
    1. partial HNSW index per dominant filter value (fastest, 0.99)
    2. relaxed_order + max_scan_tuples raised (0.76 -> 0.92)
    3. accept seqscan (planner already chooses this at low selectivity)

This module implements (2). (1) is deployment-specific and (3) is
already handled by the planner.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
from psycopg import Connection
from psycopg.rows import dict_row

from .config import RetrieverConfig, default_config
from .filters import Filters
from .fusion import Hit

logger = logging.getLogger("retriever.dense")


EMBEDDING_DIMENSIONS = 1024


# The shared SELECT body. Wrapped in a MATERIALIZED CTE only when
# relaxed_order is active (out-of-order output requires re-sort).
_SELECT_BODY = """
SELECT
    c.chunk_id,
    c.case_id,
    c.court,
    c.date,
    c.citation,
    LEFT(c.text_original, 200) AS preview,
    c.embedding <=> %(query_vec)s::halfvec(1024) AS distance
FROM public.chunks AS c
JOIN public.cases AS ca
  ON ca.case_id = c.case_id
WHERE c.is_parent = FALSE
  AND c.embedding IS NOT NULL
  AND (%(court)s::text IS NULL OR ca.court = %(court)s)
  AND (
      %(jurisdiction)s::text IS NULL
      OR ca.jurisdiction = %(jurisdiction)s
  )
  AND (
      %(published_only)s = FALSE
      OR lower(btrim(ca.precedential_status)) = 'published'
  )
  AND (%(date_from)s::date IS NULL OR ca.date >= %(date_from)s)
  AND (%(date_to)s::date IS NULL OR ca.date <= %(date_to)s)
  AND (
      %(case_ids)s::bigint[] IS NULL
      OR c.case_id = ANY(%(case_ids)s::bigint[])
  )
"""


DENSE_SQL_PLAIN = f"""
{_SELECT_BODY}
ORDER BY c.embedding <=> %(query_vec)s::halfvec(1024) ASC
LIMIT %(limit)s
"""


# The materialized CTE re-sort. `+ 0` forces the outer ORDER BY to
# execute on PG 17+, where the planner may otherwise fuse the inner
# ORDER BY into the outer query despite MATERIALIZED. It is a no-op for
# correctness (the inner result is already sorted) but protects against
# planner regressions across PG versions.
DENSE_SQL_MATERIALIZED = f"""
WITH relaxed_results AS MATERIALIZED (
    {_SELECT_BODY}
    ORDER BY c.embedding <=> %(query_vec)s::halfvec(1024) ASC
    LIMIT %(limit)s
)
SELECT
    chunk_id, case_id, court, date, citation, preview, distance
FROM relaxed_results
ORDER BY distance + 0 ASC
"""


# Backward-compatible alias for callers that imported DENSE_SQL from
# an earlier version. Points at the recommended (relaxed/materialized)
# form.
DENSE_SQL = DENSE_SQL_MATERIALIZED


def _validate_positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _vector_literal(query_vec: np.ndarray) -> tuple[str, float]:
    """Validate and serialize a 1024-dimensional vector as a text literal.

    Returns the literal and the L2 norm (for logging / sanity, unused).
    The explicit `::halfvec(1024)` cast happens in SQL. Using a text
    literal avoids depending on `register_vector()` having been called
    on this connection — the SQL never needs an adapter.
    """
    try:
        vector = np.asarray(query_vec, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("query_vec must be a numeric 1024-dimensional vector") from exc

    if vector.shape != (EMBEDDING_DIMENSIONS,):
        raise ValueError(f"query_vec must have shape ({EMBEDDING_DIMENSIONS},), got {vector.shape}")

    if not np.isfinite(vector).all():
        raise ValueError("query_vec contains NaN or infinity")

    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("query_vec must have a non-zero finite norm")

    literal = "[" + ",".join(repr(float(x)) for x in vector) + "]"
    return literal, norm


def _build_params(
    query_literal: str,
    filters: Filters,
    limit: int,
) -> dict[str, Any]:
    params = filters.sql_params()
    params.update({"query_vec": query_literal, "limit": limit})
    return params


def search(
    conn: Connection,
    query_vec: np.ndarray,
    *,
    filters: Filters | None = None,
    config: RetrieverConfig | None = None,
    limit: int | None = None,
    ef_search: int | None = None,
) -> list[Hit]:
    """Return the nearest child chunks for a 1024-dimensional query.

    Parameters
    ----------
    conn
        Open psycopg3 connection.
    query_vec
        1-D float array of length 1024. Shape, finiteness, and non-zero
        norm are validated before SQL is issued.
    filters
        Metadata restrictions. Applied inside the WHERE clause so HNSW
        sees them, not as a post-filter.
    config
        Retrieval config. Defaults to `default_config()` (env-driven).
        Controls ef_search, iterative_scan_mode, max_scan_tuples,
        scan_mem_multiplier, and the default dense_limit.
    limit
        Overrides `config.dense_limit`.
    ef_search
        Overrides `config.ef_search`. Range 1..1000.

    Returns
    -------
    list[Hit]
        Hits carry the raw cosine distance in `score`. RRF uses rank,
        not score, so distance magnitude does not affect the final
        ranking — but is preserved for debugging.

    Raises
    ------
    ValueError
        On invalid query_vec shape/values, non-positive limit, or
        ef_search out of range.
    TypeError
        On filters not being `Filters` or `None`.
    RuntimeError
        When the DB returns a non-finite distance.
    """
    if config is None:
        config = default_config()

    effective_limit = limit if limit is not None else config.dense_limit
    effective_ef_search = ef_search if ef_search is not None else config.ef_search

    _validate_positive_integer(effective_limit, "limit")
    _validate_positive_integer(effective_ef_search, "ef_search")

    if effective_ef_search > 1000:
        raise ValueError(f"ef_search must be <= 1000 (pgvector limit), got {effective_ef_search}")

    if filters is None:
        filters = Filters()
    if not isinstance(filters, Filters):
        raise TypeError("filters must be Filters or None")
    if filters.matches_nothing:
        return []

    literal, _ = _vector_literal(query_vec)
    params = _build_params(literal, filters, effective_limit)

    # SQL form depends on iterative scan mode:
    #   relaxed_order  -> out-of-order rows, CTE re-sort needed
    #   strict_order   -> ordered rows, but loses at scale (warned)
    #   off            -> ordered rows, planner may seqscan
    mode = config.iterative_scan_mode
    if mode == "relaxed_order":
        sql = DENSE_SQL_MATERIALIZED
    else:
        sql = DENSE_SQL_PLAIN
        if mode == "strict_order":
            logger.warning(
                "dense: iterative_scan_mode=strict_order — benchmarks "
                "show it loses to relaxed_order at every selectivity; "
                "use only for ablation"
            )

    with conn.transaction():
        # All four settings must be transaction-local. `set_config(...,
        # true)` is the parameterizable form of `SET LOCAL` and cannot
        # leak into the connection's session.
        conn.execute(
            "SELECT set_config('hnsw.ef_search', %s, true)",
            (str(effective_ef_search),),
        )
        conn.execute(
            "SELECT set_config('hnsw.iterative_scan', %s, true)",
            (mode,),
        )
        conn.execute(
            "SELECT set_config('hnsw.max_scan_tuples', %s, true)",
            (str(config.max_scan_tuples),),
        )
        conn.execute(
            "SELECT set_config('hnsw.scan_mem_multiplier', %s, true)",
            (str(config.scan_mem_multiplier),),
        )

        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(sql, params)
            rows = cursor.fetchall()

    # Recall detection: at 1% selectivity, relaxed_order still returns
    # the full LIMIT but a fraction of those may be wrong. Row count
    # alone cannot detect this. Log the filtered-query state at INFO so
    # eval/filtered_recall.py can correlate. Do NOT warn — the system
    # is behaving as designed; only eval can prove recall is sufficient.
    if filters.has_restrictions:
        logger.info(
            "dense: filtered query rows=%d limit=%d ef_search=%d "
            "iterative_scan=%s max_scan_tuples=%d scan_mem_multiplier=%s "
            "(recall must be verified via eval/filtered_recall.py)",
            len(rows),
            effective_limit,
            effective_ef_search,
            mode,
            config.max_scan_tuples,
            config.scan_mem_multiplier,
        )

    hits: list[Hit] = []
    for row in rows:
        distance = float(row["distance"])
        if not math.isfinite(distance):
            raise RuntimeError(
                f"database returned non-finite cosine distance for chunk {row['chunk_id']}"
            )

        row_date = row["date"]
        date_value = row_date.isoformat() if hasattr(row_date, "isoformat") else row_date

        hits.append(
            Hit(
                chunk_id=str(row["chunk_id"]),
                case_id=int(row["case_id"]),
                is_parent=False,
                # Cosine distance: smaller is better. RRF uses rank only.
                score=distance,
                rank=0,
                meta={
                    "preview": row["preview"] or "",
                    "court": row["court"],
                    "date": date_value,
                    "citation": row["citation"],
                    "retrieval_arm": "dense",
                },
            )
        )

    return hits


__all__ = [
    "EMBEDDING_DIMENSIONS",
    "DENSE_SQL",
    "DENSE_SQL_PLAIN",
    "DENSE_SQL_MATERIALIZED",
    "search",
]
