"""Exact-match and fuzzy case-name resolution for the retrieval layer.

Two resolution paths:
    exact   — case-insensitive, whitespace-normalized equality on
              cases.case_name or cases.case_name_short.
    trigram — pg_trgm similarity via the % operator when the exact
              path returns no rows.

Exact first, trigram second. A trigram match at high similarity
(default 0.60) is usually a truncated or abbreviated form of the
true name; treating it as no-match forces silent sparse-only fallback.

GIN index requirement: db.py must create an expression index on
lower(case_name) with gin_trgm_ops for the trigram SQL to use the
index rather than falling back to sequential scan.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from psycopg import Connection
from psycopg.rows import dict_row

from .filters import Filters
from .fusion import Hit

if TYPE_CHECKING:
    from .config import RetrieverConfig

logger = logging.getLogger("retriever.exact")

EXACT_CASE_NAME_SQL = """
SELECT ca.case_id
FROM public.cases AS ca
WHERE lower(btrim(coalesce(ca.case_name, ''))) = %(name_lower)s
   OR lower(btrim(coalesce(ca.case_name_short, ''))) = %(name_lower)s
ORDER BY ca.case_id ASC
LIMIT %(limit)s
"""

TRIGRAM_CASE_NAME_SQL = """
SELECT ca.case_id,
       GREATEST(
           similarity(lower(coalesce(ca.case_name, '')), %(name_lower)s),
           similarity(lower(coalesce(ca.case_name_short, '')), %(name_lower)s)
       ) AS sim
FROM public.cases AS ca
WHERE lower(coalesce(ca.case_name, '')) %% %(name_lower)s
   OR lower(coalesce(ca.case_name_short, '')) %% %(name_lower)s
ORDER BY sim DESC, ca.case_id ASC
LIMIT %(limit)s
"""

CITATION_TO_CASE_IDS_SQL = """
SELECT DISTINCT ca.case_id
FROM public.cases AS ca
WHERE lower(btrim(coalesce(ca.citation, ''))) = %(citation)s
ORDER BY ca.case_id ASC
LIMIT %(limit)s
"""

OPINION_ID_SQL = """
SELECT ca.case_id
FROM public.cases AS ca
WHERE ca.case_id = %(case_id)s
"""

CASE_IDS_TO_PARENTS_SQL = """
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


def _normalize_text(value: str) -> str:
    return " ".join(value.lower().split())


def _resolve_config(config: RetrieverConfig | None) -> RetrieverConfig:
    if config is None:
        from .config import default_config

        return default_config()
    return config


def _graph_params(filters: Filters, case_ids: list[int]) -> dict:
    params = filters.sql_params()
    user_case_filter = params.pop("case_ids")
    params["case_ids"] = case_ids
    params["filter_case_ids"] = user_case_filter
    return params


def _date_string(value) -> str | None:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return str(value.isoformat())
    return str(value)


def resolve_opinion_id(conn: Connection, opinion_id: int) -> list[int]:
    """Return [case_id] if opinion_id is known, else [].

    opinion_id corresponds to CourtListener's case_id primary key.
    """
    if not isinstance(opinion_id, int) or opinion_id <= 0:
        raise ValueError(f"opinion_id must be positive int, got {opinion_id!r}")

    with conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute(OPINION_ID_SQL, {"case_id": opinion_id})
        rows = cursor.fetchall()

    return [int(row["case_id"]) for row in rows]


def resolve_case_ids(
    conn: Connection,
    case_name: str,
    limit: int = 5,
    *,
    config: RetrieverConfig | None = None,
) -> list[int]:
    """Resolve a case name to candidate case IDs.

    Tries exact match first. Falls back to trigram similarity at
    config.trigram_similarity_threshold (default 0.60) only if exact
    match returns zero rows.

    Returns [] if neither path finds a match. Callers must fail closed.
    """
    if not isinstance(case_name, str) or not case_name.strip():
        return []
    if not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be positive int")

    cfg = _resolve_config(config)
    name_lower = _normalize_text(case_name)

    with conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute(EXACT_CASE_NAME_SQL, {"name_lower": name_lower, "limit": limit})
        exact_rows = cursor.fetchall()

    exact_ids = [int(row["case_id"]) for row in exact_rows]
    if exact_ids:
        logger.debug("resolve_case_ids: exact hit for %r -> %s", case_name, exact_ids)
        return exact_ids

    threshold = float(cfg.trigram_similarity_threshold)
    with conn.transaction():
        conn.execute(
            "SELECT set_config('pg_trgm.similarity_threshold', %s, true)",
            (str(threshold),),
        )
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(TRIGRAM_CASE_NAME_SQL, {"name_lower": name_lower, "limit": limit})
            trigram_rows = cursor.fetchall()

    trigram_ids = [int(row["case_id"]) for row in trigram_rows]

    if trigram_ids:
        logger.info(
            "resolve_case_ids: trigram fallback for %r (threshold=%.2f) -> %s",
            case_name,
            threshold,
            trigram_ids,
        )
    else:
        logger.debug(
            "resolve_case_ids: no match for %r (exact=0, trigram=0 at threshold=%.2f)",
            case_name,
            threshold,
        )

    return trigram_ids


def search(
    conn: Connection,
    citation: str,
    *,
    limit: int = 20,
    filters: Filters | None = None,
    config: RetrieverConfig | None = None,
) -> list[Hit]:
    """Resolve a reporter citation to parent Hit objects.

    Citation is normalized (lowercase + collapsed whitespace) and matched
    against cases.citation. Matches are expanded to one parent hit per
    case, respecting filters. Score is left at 0.0 — RRF uses rank, not
    raw score, for citation matches where every hit is equally valid.

    Returns [] if no case carries that citation.
    """
    if not isinstance(citation, str) or not citation.strip():
        return []
    if not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be positive int")

    if filters is None:
        filters = Filters()
    if not isinstance(filters, Filters):
        raise TypeError("filters must be Filters or None")
    if filters.matches_nothing:
        return []

    citation_lower = _normalize_text(citation)

    with conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute(CITATION_TO_CASE_IDS_SQL, {"citation": citation_lower, "limit": limit})
        rows = cursor.fetchall()

    case_ids = [int(row["case_id"]) for row in rows]
    if not case_ids:
        logger.debug("exact.search: no case for citation %r", citation)
        return []

    return case_ids_to_hits(conn, case_ids, filters=filters)


def case_ids_to_hits(
    conn: Connection,
    case_ids: list[int],
    *,
    filters: Filters | None = None,
) -> list[Hit]:
    """Expand case IDs into one parent Hit per case.

    Parent selection (per case, fixed):
        highest section_priority, then lowest section_index, then
        lowest chunk_id.

    Return order follows input case_ids order.
    """
    if not case_ids:
        return []

    if filters is None:
        filters = Filters()
    if not isinstance(filters, Filters):
        raise TypeError("filters must be Filters or None")
    if filters.matches_nothing:
        return []

    params = _graph_params(filters, case_ids)

    with conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute(CASE_IDS_TO_PARENTS_SQL, params)
        rows = cursor.fetchall()

    hits = [
        Hit(
            chunk_id=str(row["chunk_id"]),
            case_id=int(row["case_id"]),
            is_parent=True,
            score=0.0,
            rank=0,
            meta={
                "preview": row["preview"] or "",
                "court": row["court"],
                "date": _date_string(row["date"]),
                "citation": row["citation"],
                "retrieval_arm": "exact",
            },
        )
        for row in rows
    ]

    order = {case_id: i for i, case_id in enumerate(case_ids)}
    hits.sort(key=lambda hit: order.get(hit.case_id, len(case_ids)))
    return hits


__all__ = [
    "resolve_opinion_id",
    "resolve_case_ids",
    "search",
    "case_ids_to_hits",
    "EXACT_CASE_NAME_SQL",
    "TRIGRAM_CASE_NAME_SQL",
    "CITATION_TO_CASE_IDS_SQL",
    "OPINION_ID_SQL",
    "CASE_IDS_TO_PARENTS_SQL",
]
