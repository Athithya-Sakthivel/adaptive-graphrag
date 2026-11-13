"""Database source for the case-law gold-set builder.

Implements the eval-framework source contract:
    corpus_case_count() -> int
    corpus_fingerprint() -> str
    load_candidates() -> list[dict]
    selection_sql() -> str
    close()

Candidates are cases with the metadata needed to generate HOLDING,
LOOKUP, and VALIDITY queries. Overruling citers are detected by a
text heuristic against the citing case's parent text — this is
imperfect and the manifest records `heuristic` so eval results
carry that caveat.
"""

from __future__ import annotations

import os
from typing import Any

import psycopg
from psycopg.rows import dict_row

from .fingerprint import case_law_fingerprint

_OVERRULE_STEMS = ("overrul", "revers", "vacat")

_SELECTION_SQL = """
WITH citer_counts AS (
    SELECT target_id AS case_id, COUNT(DISTINCT source_id) AS indegree
    FROM case_edges
    WHERE relation = 'CITES'
    GROUP BY target_id
),
best_holding AS (
    SELECT DISTINCT ON (chunk_id_case)
        chunk_id_case,
        chunk_id,
        text_original
    FROM (
        SELECT
            case_id AS chunk_id_case,
            chunk_id,
            text_original,
            section_priority,
            section_index
        FROM chunks
        WHERE is_parent = TRUE AND section = 'holding'
    ) h
    ORDER BY chunk_id_case, section_priority DESC, section_index ASC, chunk_id ASC
)
SELECT
    ca.case_id,
    ca.case_name,
    ca.court,
    EXTRACT(YEAR FROM ca.date)::int AS year,
    COALESCE(cc.indegree, 0) AS indegree,
    bh.chunk_id AS holding_chunk_id,
    LEFT(bh.text_original, 600) AS holding_text
FROM cases ca
LEFT JOIN citer_counts cc ON cc.case_id = ca.case_id
LEFT JOIN best_holding bh ON bh.chunk_id_case = ca.case_id
WHERE ca.case_name IS NOT NULL AND ca.case_name <> ''
ORDER BY ca.case_id
"""


class EvalSource:
    """Database-backed source of gold-set candidates."""

    def __init__(self) -> None:
        dsn = os.environ.get("DATABASE_URL")
        if not dsn:
            raise RuntimeError("DATABASE_URL must be set")
        self.conn = psycopg.connect(dsn, autocommit=True)
        self._corpus_hash = case_law_fingerprint(self.conn)

    def corpus_case_count(self) -> int:
        with self.conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM cases")
            return int(cur.fetchone()[0])

    def corpus_fingerprint(self) -> str:
        return self._corpus_hash

    def selection_sql(self) -> str:
        return _SELECTION_SQL

    def load_candidates(self) -> list[dict]:
        """Load all candidate cases with their citers and holding chunks."""
        with self.conn.cursor(row_factory=dict_row) as cur:
            cur.execute(_SELECTION_SQL)
            rows = cur.fetchall()

        case_ids = [int(row["case_id"]) for row in rows]
        citers_by_case = self._fetch_citers(case_ids)
        overruling_by_case = self._fetch_overruling_citers(case_ids)

        candidates: list[dict[str, Any]] = []
        for row in rows:
            case_id = int(row["case_id"])
            candidates.append(
                {
                    "case_id": case_id,
                    "case_name": str(row["case_name"]),
                    "court": str(row["court"] or "unknown"),
                    "year": int(row["year"] or 1900),
                    "indegree": int(row["indegree"] or 0),
                    "holding_chunk_id": row["holding_chunk_id"],
                    "holding_text": row["holding_text"] or "",
                    "citer_case_ids": citers_by_case.get(case_id, []),
                    "overruling_citer_ids": overruling_by_case.get(case_id, []),
                }
            )

        return candidates

    def _fetch_citers(self, case_ids: list[int]) -> dict[int, list[int]]:
        if not case_ids:
            return {}
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT target_id, array_agg(DISTINCT source_id ORDER BY source_id)
                FROM case_edges
                WHERE relation = 'CITES' AND target_id = ANY(%s::bigint[])
                GROUP BY target_id
                """,
                (case_ids,),
            )
            return {int(tgt): [int(s) for s in srcs] for tgt, srcs in cur.fetchall()}

    def _fetch_overruling_citers(self, case_ids: list[int]) -> dict[int, list[int]]:
        """Find citers whose parent chunk contains overruling language.

        Heuristic: case-insensitive substring match on stems 'overrul',
        'revers', 'vacat' in the citing case's parent chunk text.
        Imperfect — misses implicit overrulings and over-reports
        cases that merely discuss reversal of a different holding.
        """
        if not case_ids:
            return {}
        stems = " OR ".join(f"lower(c.text_original) LIKE '%%{stem}%%'" for stem in _OVERRULE_STEMS)
        sql = f"""
            SELECT e.target_id, array_agg(DISTINCT e.source_id ORDER BY e.source_id)
            FROM case_edges e
            JOIN chunks c ON c.case_id = e.source_id AND c.is_parent
            WHERE e.relation = 'CITES'
              AND e.target_id = ANY(%s::bigint[])
              AND ({stems})
            GROUP BY e.target_id
        """
        with self.conn.cursor() as cur:
            cur.execute(sql, (case_ids,))
            return {int(tgt): [int(s) for s in srcs] for tgt, srcs in cur.fetchall()}

    def close(self) -> None:
        self.conn.close()


def build_source() -> EvalSource:
    """Factory called by custom_eval_builder."""
    return EvalSource()


__all__ = ["EvalSource", "build_source"]
