"""Exact-match and fuzzy case-name resolution for the retrieval layer.

Two resolution paths:
    exact     — case-insensitive, whitespace-normalized equality on
                ``cases.case_name`` or ``cases.case_name_short``.
    trigram   — pg_trgm similarity via the ``%`` operator when the
                exact path returns no rows.

Exact first, trigram second. A trigram match at high similarity (0.60
default, configurable via ``RetrieverConfig.trigram_similarity_threshold``)
is usually a truncated or abbreviated form of the true name — e.g.
"Andrews v. Merritt Oil" vs "Andrews v. Merritt Oil Co., Inc." — and
treating it as no-match forces the router into silent sparse-only
fallback. Trigram resolution is the lesser evil; the router still
fails closed when *neither* path finds a case.

GIN index note
--------------
The trigram SQL uses ``lower(case_name) % name_lower`` for case
insensitivity. For the GIN index to be used, ``db.py`` must create an
expression index on ``lower(case_name)``::

    CREATE INDEX CONCURRENTLY idx_cases_case_name_trgm
    ON public.cases USING gin (lower(case_name) gin_trgm_ops)

If the index is instead defined on ``case_name`` (without ``lower``),
the SQL remains correct but falls back to a sequential scan. At 1,553
cases this is fine; at scale it is not.

The ``%`` operator (not ``similarity()`` in WHERE) is the index-
eligible form. Writing ``WHERE similarity(a, b) >= x`` forces a full
scan regardless of any index. ``similarity()`` appears only in ORDER
BY, re-scoring the small candidate set the index already filtered.

Threshold scope
---------------
``pg_trgm.similarity_threshold`` is set via
``set_config(..., true)`` — transaction-local, so the caller's session
state is restored when the query transaction ends. A per-connection
shared value would leak between unrelated queries; this avoids that.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from psycopg import Connection
from psycopg.rows import dict_row

from .filters import Filters
from .fusion import Hit
from .graph import case_ids_to_hits

if TYPE_CHECKING:
    from .config import RetrieverConfig


logger = logging.getLogger("retriever.exact")


# ── SQL ───────────────────────────────────────────────────────────────

# Exact case-name equality (case-insensitive, whitespace-normalized).
# Checks both the full name and the short name; either match counts.
EXACT_CASE_NAME_SQL = """
SELECT ca.case_id
FROM public.cases AS ca
WHERE lower(btrim(coalesce(ca.case_name, ''))) = %(name_lower)s
   OR lower(btrim(coalesce(ca.case_name_short, ''))) = %(name_lower)s
ORDER BY ca.case_id ASC
LIMIT %(limit)s
"""

# Trigram fuzzy match. The ``%%`` sequences are psycopg escapes for a
# literal ``%`` (the pg_trgm operator), and ``%(name_lower)s`` /
# ``%(limit)s`` are named parameters. Only ``similarity()`` — used for
# ordering — appears in SELECT, never in WHERE; putting it in WHERE
# would bypass the GIN index.
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

# Reporter-citation resolution. CourtListener stores citations without
# a year suffix; a query like "410 U.S. 113" matches exactly. Queries
# with year suffixes are handled by the caller (classify.extract_citation
# returns just the reporter portion).
CITATION_TO_CASE_IDS_SQL = """
SELECT DISTINCT ca.case_id
FROM public.cases AS ca
WHERE lower(btrim(coalesce(ca.citation, ''))) = %(citation)s
ORDER BY ca.case_id ASC
LIMIT %(limit)s
"""

# Opinion-ID resolution. ``cases.case_id`` is CourtListener's opinion
# ID; a direct primary-key lookup.
OPINION_ID_SQL = """
SELECT ca.case_id
FROM public.cases AS ca
WHERE ca.case_id = %(case_id)s
"""


# ── Helpers ───────────────────────────────────────────────────────────


def _normalize_case_name(case_name: str) -> str:
    """Lowercase, trim, collapse internal whitespace."""
    return " ".join(case_name.lower().split())


def _normalize_citation(citation: str) -> str:
    """Lowercase, trim, collapse internal whitespace."""
    return " ".join(citation.lower().split())


def _resolve_config(config: RetrieverConfig | None) -> RetrieverConfig:
    if config is None:
        from .config import default_config

        return default_config()
    return config


# ── Public API ────────────────────────────────────────────────────────


def resolve_opinion_id(conn: Connection, opinion_id: int) -> list[int]:
    """Return ``[case_id]`` if ``opinion_id`` is a known case, else ``[]``.

    ``opinion_id`` corresponds to CourtListener's ``case_id`` primary
    key. The return is a list for uniformity with ``resolve_case_ids``;
    it has at most one element.
    """
    if isinstance(opinion_id, bool) or not isinstance(opinion_id, int):
        raise TypeError(f"opinion_id must be an integer, got {opinion_id!r}")
    if opinion_id <= 0:
        raise ValueError(f"opinion_id must be positive, got {opinion_id}")

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

    Tries exact match first. Only if that returns zero rows does it
    fall back to a trigram similarity query at
    ``config.trigram_similarity_threshold`` (default 0.60).

    Returns ``[]`` if neither path finds a match. Callers (router) must
    fail closed — never graph-traverse from semantically similar but
    wrong seeds.
    """
    if not isinstance(case_name, str):
        raise TypeError(f"case_name must be a string, got {type(case_name).__name__}")
    if not case_name.strip():
        return []
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be a positive integer")

    cfg = _resolve_config(config)
    name_lower = _normalize_case_name(case_name)

    # ---- Path 1: exact match ----
    with conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute(
            EXACT_CASE_NAME_SQL,
            {"name_lower": name_lower, "limit": limit},
        )
        exact_rows = cursor.fetchall()

    exact_ids = [int(row["case_id"]) for row in exact_rows]
    if exact_ids:
        logger.debug(
            "resolve_case_ids: exact hit for %r → %s",
            case_name,
            exact_ids,
        )
        return exact_ids

    # ---- Path 2: trigram fallback ----
    threshold = float(cfg.trigram_similarity_threshold)
    with conn.transaction():
        # set_config(..., true) = transaction-local: restores the
        # caller's session value when this transaction commits.
        conn.execute(
            "SELECT set_config('pg_trgm.similarity_threshold', %s, true)",
            (str(threshold),),
        )
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                TRIGRAM_CASE_NAME_SQL,
                {"name_lower": name_lower, "limit": limit},
            )
            trigram_rows = cursor.fetchall()

    trigram_ids = [int(row["case_id"]) for row in trigram_rows]

    if trigram_ids:
        logger.info(
            "resolve_case_ids: trigram fallback for %r (threshold=%.2f) → %s",
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
    """Resolve a reporter citation to parent ``Hit`` objects.

    The citation string is normalized (lowercase + collapsed whitespace)
    and matched against ``cases.citation``. Matches are expanded to one
    parent hit per case, respecting ``filters``. Score is left at the
    default (0.0) — RRF uses rank, not raw score, for citation matches
    where every hit is equally valid.

    Returns ``[]`` if no case carries that citation or if
    ``filters.matches_nothing`` is True.
    """
    if not isinstance(citation, str):
        raise TypeError(f"citation must be a string, got {type(citation).__name__}")
    if not citation.strip():
        return []
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be a positive integer")

    if filters is None:
        filters = Filters()
    if not isinstance(filters, Filters):
        raise TypeError("filters must be Filters or None")

    if filters.matches_nothing:
        return []

    cfg = _resolve_config(config)
    citation_lower = _normalize_citation(citation)

    with conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute(
            CITATION_TO_CASE_IDS_SQL,
            {"citation": citation_lower, "limit": limit},
        )
        rows = cursor.fetchall()

    case_ids = [int(row["case_id"]) for row in rows]
    if not case_ids:
        logger.debug("exact.search: no case for citation %r", citation)
        return []

    hits = case_ids_to_hits(
        conn,
        case_ids,
        filters=filters,
        config=cfg,
    )

    logger.debug(
        "exact.search: citation=%r → %d cases, %d hits",
        citation,
        len(case_ids),
        len(hits),
    )

    return hits


__all__ = [
    "resolve_opinion_id",
    "resolve_case_ids",
    "search",
    "EXACT_CASE_NAME_SQL",
    "TRIGRAM_CASE_NAME_SQL",
    "CITATION_TO_CASE_IDS_SQL",
    "OPINION_ID_SQL",
]
