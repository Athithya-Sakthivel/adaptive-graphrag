"""Sparse legal-text retrieval using PostgreSQL tsvector and GIN.

Uses ts_rank_cd for proximity-based ranking of lexical matches.
ts_rank_cd is not BM25 — it does not provide inverse-document-frequency
weighting or term-frequency saturation. It is a useful built-in
lexical ranker, not a substitute for a true BM25 extension.

The query is parsed once in a materialized CTE to avoid redundant
websearch_to_tsquery computation during the main scan.
"""

from __future__ import annotations

from typing import Any

from psycopg import Connection
from psycopg.rows import dict_row

from .filters import Filters
from .fusion import Hit

SPARSE_SQL = """
WITH parsed_query AS MATERIALIZED (
    SELECT websearch_to_tsquery('english', %(query)s) AS tsq
)
SELECT
    c.chunk_id,
    c.case_id,
    c.court,
    c.date,
    c.citation,
    LEFT(c.text_original, 200) AS preview,
    ts_rank_cd(c.tsv, parsed_query.tsq) AS lex
FROM public.chunks AS c
JOIN public.cases AS ca ON ca.case_id = c.case_id
CROSS JOIN parsed_query
WHERE c.is_parent = FALSE
  AND c.tsv @@ parsed_query.tsq
  AND (%(court)s::text IS NULL OR ca.court = %(court)s)
  AND (%(jurisdiction)s::text IS NULL OR ca.jurisdiction = %(jurisdiction)s)
  AND (%(published_only)s = FALSE OR lower(btrim(ca.precedential_status)) = 'published')
  AND (%(date_from)s::date IS NULL OR ca.date >= %(date_from)s)
  AND (%(date_to)s::date IS NULL OR ca.date <= %(date_to)s)
  AND (%(case_ids)s::bigint[] IS NULL OR c.case_id = ANY(%(case_ids)s::bigint[]))
ORDER BY lex DESC, c.chunk_id ASC
LIMIT %(limit)s
"""


def _validate_limit(limit: int) -> None:
    if not isinstance(limit, int) or limit < 1:
        raise ValueError(f"limit must be positive int, got {limit!r}")


def search(
    conn: Connection,
    query_text: str,
    *,
    filters: Filters | None = None,
    limit: int = 50,
) -> list[Hit]:
    """Search child chunks using PostgreSQL full-text matching.

    Empty and whitespace-only queries return [] without touching SQL.

    Returns:
        List of Hit objects with ts_rank_cd in score field (higher=better).
        RRF uses rank, not score, so magnitude doesn't affect fusion.
    """
    if not isinstance(query_text, str):
        raise TypeError(f"query_text must be str, got {type(query_text).__name__}")

    query_text = query_text.strip()
    if not query_text:
        return []

    if len(query_text) > 10000:
        raise ValueError("query_text exceeds 10,000-character safety limit")

    _validate_limit(limit)

    if filters is None:
        filters = Filters()
    if not isinstance(filters, Filters):
        raise TypeError("filters must be Filters or None")
    if filters.matches_nothing:
        return []

    params: dict[str, Any] = filters.sql_params()
    params.update({"query": query_text, "limit": limit})

    with conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute(SPARSE_SQL, params)
        rows = cursor.fetchall()

    hits: list[Hit] = []
    for row in rows:
        score = float(row["lex"])
        if score < 0:
            raise RuntimeError(f"negative lexical score for chunk {row['chunk_id']}")
        if not (score == score and abs(score) != float("inf")):
            raise RuntimeError(f"non-finite lexical score for chunk {row['chunk_id']}")

        row_date = row["date"]
        date_value = row_date.isoformat() if hasattr(row_date, "isoformat") else row_date

        hits.append(
            Hit(
                chunk_id=str(row["chunk_id"]),
                case_id=int(row["case_id"]),
                is_parent=False,
                score=score,
                rank=0,
                meta={
                    "preview": row["preview"] or "",
                    "court": row["court"],
                    "date": date_value,
                    "citation": row["citation"],
                    "retrieval_arm": "sparse",
                },
            )
        )

    return hits


__all__ = [
    "SPARSE_SQL",
    "search",
]
