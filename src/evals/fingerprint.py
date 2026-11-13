"""Canonical corpus fingerprint for eval reproducibility.

The fingerprint is a SHA-256 digest over the complete state of the
indexed corpus: chunker versions, chunk IDs, case IDs, and citation
edges. If any of these change (re-chunking, new data, edge filter
change), the fingerprint changes and the eval runner will refuse to
score against a gold set built on a different corpus state.

This prevents silent "chunker drift" — measuring retrieval quality
against a gold set whose chunk boundaries no longer match the index.
"""

from __future__ import annotations

import hashlib

from psycopg import Connection

_SCHEMA_VERSION = "case-law-v1"


def case_law_fingerprint(conn: Connection) -> str:
    """Compute SHA-256 over canonical case-law corpus state.

    Deterministic given the same database state. Costs ~1-2s on a
    17K-chunk corpus; run once per eval session.

    Drift detected in:
        - chunker_version (any re-chunk flips the hash)
        - chunk_id set (encodes section + index)
        - case_id set
        - citation edge set (source, target, relation)
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


__all__ = ["case_law_fingerprint"]
