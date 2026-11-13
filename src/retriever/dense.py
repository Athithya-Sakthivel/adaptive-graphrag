"""Dense retrieval using PostgreSQL pgvector HNSW with halfvec(384).

Uses relaxed_order iterative scanning with materialized CTE re-sort
to preserve distance ordering while maintaining recall under filters.
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

EMBEDDING_DIMENSIONS = 384

_SELECT_BODY = """
SELECT
    c.chunk_id,
    c.case_id,
    c.court,
    c.date,
    c.citation,
    LEFT(c.text_original, 200) AS preview,
    c.embedding <=> %(query_vec)s::halfvec(384) AS distance
FROM public.chunks AS c
JOIN public.cases AS ca ON ca.case_id = c.case_id
WHERE c.is_parent = FALSE
  AND c.embedding IS NOT NULL
  AND (%(court)s::text IS NULL OR ca.court = %(court)s)
  AND (%(jurisdiction)s::text IS NULL OR ca.jurisdiction = %(jurisdiction)s)
  AND (%(published_only)s = FALSE OR lower(btrim(ca.precedential_status)) = 'published')
  AND (%(date_from)s::date IS NULL OR ca.date >= %(date_from)s)
  AND (%(date_to)s::date IS NULL OR ca.date <= %(date_to)s)
  AND (%(case_ids)s::bigint[] IS NULL OR c.case_id = ANY(%(case_ids)s::bigint[]))
"""

DENSE_SQL_PLAIN = f"""
{_SELECT_BODY}
ORDER BY c.embedding <=> %(query_vec)s::halfvec(384) ASC
LIMIT %(limit)s
"""

DENSE_SQL_MATERIALIZED = f"""
WITH relaxed_results AS MATERIALIZED (
    {_SELECT_BODY}
    ORDER BY c.embedding <=> %(query_vec)s::halfvec(384) ASC
    LIMIT %(limit)s
)
SELECT chunk_id, case_id, court, date, citation, preview, distance
FROM relaxed_results
ORDER BY distance + 0 ASC
"""

DENSE_SQL = DENSE_SQL_MATERIALIZED


def _validate_positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be positive int")


def _vector_literal(query_vec: np.ndarray) -> tuple[str, float]:
    """Serialize 384-dim vector as text literal, return (literal, norm)."""
    try:
        vector = np.asarray(query_vec, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("query_vec must be numeric 384-dim vector") from exc

    if vector.shape != (EMBEDDING_DIMENSIONS,):
        raise ValueError(f"query_vec must have shape ({EMBEDDING_DIMENSIONS},), got {vector.shape}")

    if not np.isfinite(vector).all():
        raise ValueError("query_vec contains NaN or infinity")

    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("query_vec must have non-zero finite norm")

    literal = "[" + ",".join(repr(float(x)) for x in vector) + "]"
    return literal, norm


def _build_params(query_literal: str, filters: Filters, limit: int) -> dict[str, Any]:
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
    """Return nearest child chunks for a 384-dim query vector.

    Args:
        conn: Open psycopg connection.
        query_vec: 1-D float array of length 384.
        filters: Metadata restrictions (applied inside WHERE clause).
        config: Retrieval config (defaults to default_config()).
        limit: Override config.dense_limit.
        ef_search: Override config.ef_search (range 1..1000).

    Returns:
        List of Hit objects with cosine distance in score field.
    """
    if config is None:
        config = default_config()

    effective_limit = limit if limit is not None else config.dense_limit
    effective_ef_search = ef_search if ef_search is not None else config.ef_search

    _validate_positive_integer(effective_limit, "limit")
    _validate_positive_integer(effective_ef_search, "ef_search")

    if effective_ef_search > 1000:
        raise ValueError(f"ef_search must be <= 1000, got {effective_ef_search}")

    if filters is None:
        filters = Filters()
    if not isinstance(filters, Filters):
        raise TypeError("filters must be Filters or None")
    if filters.matches_nothing:
        return []

    literal, _ = _vector_literal(query_vec)
    params = _build_params(literal, filters, effective_limit)

    mode = config.iterative_scan_mode
    if mode == "relaxed_order":
        sql = DENSE_SQL_MATERIALIZED
    else:
        sql = DENSE_SQL_PLAIN
        if mode == "strict_order":
            logger.warning(
                "dense: iterative_scan_mode=strict_order loses recall at scale; "
                "use relaxed_order for production"
            )

    with conn.transaction():
        conn.execute("SELECT set_config('hnsw.ef_search', %s, true)", (str(effective_ef_search),))
        conn.execute("SELECT set_config('hnsw.iterative_scan', %s, true)", (mode,))
        conn.execute(
            "SELECT set_config('hnsw.max_scan_tuples', %s, true)", (str(config.max_scan_tuples),)
        )
        conn.execute(
            "SELECT set_config('hnsw.scan_mem_multiplier', %s, true)",
            (str(config.scan_mem_multiplier),),
        )

        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(sql, params)
            rows = cursor.fetchall()

    if filters.has_restrictions:
        logger.info(
            "dense: filtered query rows=%d limit=%d ef_search=%d iterative_scan=%s",
            len(rows),
            effective_limit,
            effective_ef_search,
            mode,
        )

    hits: list[Hit] = []
    for row in rows:
        distance = float(row["distance"])
        if not math.isfinite(distance):
            raise RuntimeError(f"non-finite cosine distance for chunk {row['chunk_id']}")

        row_date = row["date"]
        date_value = row_date.isoformat() if hasattr(row_date, "isoformat") else row_date

        hits.append(
            Hit(
                chunk_id=str(row["chunk_id"]),
                case_id=int(row["case_id"]),
                is_parent=False,
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
