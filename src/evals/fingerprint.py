"""Canonical corpus fingerprints for eval reproducibility.

The same fingerprint implementation must be used by both the gold builder
(`eval_source.py`) and the eval runtime (`eval_adapter.py`). If they
diverge, `run_eval.py` refuses to score — and a silent divergence means
you'd be measuring "chunker drift" not "retrieval quality".

LegalBench uses a separate fingerprint (`legalbench_fingerprint`) because
its corpus is contract text, not case law.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import psycopg

_SCHEMA_VERSION = "case-law-v1"


def case_law_fingerprint(conn: psycopg.Connection) -> str:
    """SHA-256 over the canonical case-law corpus state.

    Detects drift in: chunker versions, chunk IDs (which encode section +
    index, so a re-chunk flips the hash), case IDs, and citation edges.

    Costs ~1-2s on a 17K-chunk corpus. Run once per eval session.
    """
    digest = hashlib.sha256()
    digest.update(f"schema={_SCHEMA_VERSION}\0".encode())

    with conn.cursor() as cur:
        cur.execute(
            "SELECT chunker_version, COUNT(*) FROM chunks "
            "GROUP BY chunker_version ORDER BY chunker_version"
        )
        for version, count in cur.fetchall():
            digest.update(f"chunker={version or '<null>'}\0n={count}\0".encode())

        cur.execute("SELECT chunk_id FROM chunks ORDER BY chunk_id")
        for (chunk_id,) in cur.fetchall():
            digest.update(f"chunk={chunk_id}\0".encode())

        cur.execute("SELECT case_id FROM cases ORDER BY case_id")
        for (case_id,) in cur.fetchall():
            digest.update(f"case={case_id}\0".encode())

        cur.execute(
            "SELECT source_id, target_id, relation FROM case_edges "
            "ORDER BY source_id, target_id, relation"
        )
        for source, target, relation in cur.fetchall():
            digest.update(f"edge={source}:{target}:{relation}\0".encode())

    return digest.hexdigest()


def legalbench_fingerprint(corpus_root: Path) -> str:
    """LegalBench corpus fingerprint (independent of case-law)."""
    from evals.legalbench_rag import corpus_fingerprint

    return corpus_fingerprint(corpus_root)
