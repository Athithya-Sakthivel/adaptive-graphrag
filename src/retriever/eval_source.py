"""Database source for the custom case-law gold builder.

Implements the eval-framework contract:

    source.corpus_case_count() -> int
    source.corpus_fingerprint() -> str
    source.load_candidates() -> list[dict]
    source.selection_sql() -> str
    source.close()

Candidate fields (per row):

    case_id, case_name, court, year, indegree,
    holding_chunk_id, holding_text,
    citer_case_ids, overruling_citer_ids, application_case_ids,
    doctrine, doctrine_leading

Overruling/application citers are detected by a text heuristic against
the citing case's parent text. This is imperfect — it will miss
overrulings phrased differently and it will over-report cases that
merely mention a doctrine. The manifest records `heuristic` so eval
results can be interpreted with that caveat.

Doctrine detection uses a fixed list of common legal doctrines. Cases
whose holding mentions a doctrine are doctrine candidates; among
those, high-indegree cases are considered "leading" for that doctrine.
"""

from __future__ import annotations

import os
from typing import Any

import psycopg
from psycopg.rows import dict_row

from evals.fingerprint import case_law_fingerprint

_DOCTRINES = (
    "qualified immunity",
    "summary judgment",
    "due process",
    "equal protection",
    "probable cause",
    "burden of proof",
    "res judicata",
    "collateral estoppel",
    "statute of limitations",
    "contributory negligence",
    "promissory estoppel",
    "good faith",
    "bad faith",
    "negligence per se",
    "strict liability",
    "excessive force",
    "motion to dismiss",
    "motion in limine",
    "directed verdict",
    "jury instruction",
)

_LEADING_DOCTRINE_MIN_DEGREE = 5

_OVERRULE_STEMS = ("overrul", "revers", "vacat")


def _detect_doctrine(holding_text: str) -> str | None:
    """Return the first doctrine keyword found in the holding text."""
    lower = holding_text.lower()
    for doctrine in _DOCTRINES:
        if doctrine in lower:
            return doctrine
    return None


_SELECTION_SQL = """
-- Candidate rows for gold construction.
-- One row per case with citers-count and best holding chunk.
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
        with self.conn.cursor(row_factory=dict_row) as cur:
            cur.execute(_SELECTION_SQL)
            rows = cur.fetchall()

        case_ids = [int(row["case_id"]) for row in rows]

        # --- Citers per case (batch) ---
        citers_by_case = self._fetch_citers(case_ids)

        # --- Overruling citers (batch) ---
        overruling_by_case = self._fetch_overruling_citers(case_ids)

        # --- Doctrine per case, then application citers per doctrine ---
        doctrines_by_case: dict[int, str] = {}
        for row in rows:
            text = row["holding_text"] or ""
            if text:
                d = _detect_doctrine(text)
                if d is not None:
                    doctrines_by_case[int(row["case_id"])] = d

        application_by_case = self._fetch_application_citers(doctrines_by_case)

        candidates: list[dict[str, Any]] = []
        for row in rows:
            case_id = int(row["case_id"])
            indegree = int(row["indegree"] or 0)
            doctrine = doctrines_by_case.get(case_id)

            candidates.append(
                {
                    "case_id": case_id,
                    "case_name": str(row["case_name"]),
                    "court": str(row["court"] or "unknown"),
                    "year": int(row["year"] or 1900),
                    "indegree": indegree,
                    "holding_chunk_id": row["holding_chunk_id"],
                    "holding_text": row["holding_text"] or "",
                    "citer_case_ids": citers_by_case.get(case_id, []),
                    "overruling_citer_ids": overruling_by_case.get(case_id, []),
                    "application_case_ids": application_by_case.get(case_id, []),
                    "doctrine": doctrine,
                    "doctrine_leading": (
                        doctrine is not None and indegree >= _LEADING_DOCTRINE_MIN_DEGREE
                    ),
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
            return {int(target): [int(s) for s in sources] for target, sources in cur.fetchall()}

    def _fetch_overruling_citers(self, case_ids: list[int]) -> dict[int, list[int]]:
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
            return {int(target): [int(s) for s in sources] for target, sources in cur.fetchall()}

    def _fetch_application_citers(self, doctrines_by_case: dict[int, str]) -> dict[int, list[int]]:
        """For each case with a detected doctrine, find citers mentioning it."""
        if not doctrines_by_case:
            return {}

        # Group by doctrine to batch SQL (max ~20 doctrines)
        by_doctrine: dict[str, list[int]] = {}
        for case_id, doctrine in doctrines_by_case.items():
            by_doctrine.setdefault(doctrine, []).append(case_id)

        result: dict[int, list[int]] = {}
        with self.conn.cursor() as cur:
            for doctrine, targets in by_doctrine.items():
                cur.execute(
                    """
                    SELECT e.target_id, array_agg(DISTINCT e.source_id ORDER BY e.source_id)
                    FROM case_edges e
                    JOIN chunks c ON c.case_id = e.source_id AND c.is_parent
                    WHERE e.relation = 'CITES'
                      AND e.target_id = ANY(%s::bigint[])
                      AND lower(c.text_original) LIKE '%%' || %s || '%%'
                    GROUP BY e.target_id
                    """,
                    (targets, doctrine.lower()),
                )
                for target, sources in cur.fetchall():
                    result.setdefault(int(target), []).extend(int(s) for s in sources)

        # Deduplicate
        return {k: list(dict.fromkeys(v)) for k, v in result.items()}

    def close(self) -> None:
        self.conn.close()


def build_source() -> EvalSource:
    return EvalSource()
