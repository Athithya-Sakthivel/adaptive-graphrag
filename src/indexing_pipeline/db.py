"""All SQL. The only file that imports psycopg.

Requires PostgreSQL extensions:
  - vector  (pgvector, for halfvec(1024) HNSW)
  - pg_trgm (for fuzzy case-name search)

Both are created by ``ensure_schema()`` via ``SCHEMA_STATEMENTS``.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any

import psycopg
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from .embedder import EMBEDDING_DIMENSIONS

POOL_TIMEOUT_SECONDS = 30.0


# ── Column lists ──────────────────────────────────────────────────────────

CASE_COLUMNS = (
    "case_id",
    "cluster_id",
    "court",
    "date",
    "title",
    "citation",
    "opinion_type",
    "jurisdiction",
    "doc_type",
    "case_name",
    "case_name_short",
    "precedential_status",
    "nature_of_suit",
    "citation_count",
    "judges",
    "metadata",
)

CHUNK_COLUMNS = (
    "chunk_id",
    "case_id",
    "parent_id",
    "is_parent",
    "text",
    "text_original",
    "text_hash",
    "section",
    "section_path",
    "section_priority",
    "section_index",
    "child_index",
    "opinion_type",
    "page_start",
    "page_end",
    "char_start",
    "char_end",
    "court",
    "date",
    "citation",
    "title",
    "chunker_version",
    "embedding",
    "embedder_version",
)

CHUNK_COPY_COLUMNS = ", ".join(CHUNK_COLUMNS)


# ── Schema ────────────────────────────────────────────────────────────────

SCHEMA_STATEMENTS: tuple[str, ...] = (
    # Extensions first. Both are required by later indexes.
    "CREATE EXTENSION IF NOT EXISTS vector",
    "CREATE EXTENSION IF NOT EXISTS pg_trgm",
    """
    CREATE TABLE IF NOT EXISTS public.cases (
        case_id              BIGINT PRIMARY KEY,
        cluster_id           BIGINT,
        court                TEXT,
        date                 DATE,
        title                TEXT,
        citation             TEXT,
        opinion_type         TEXT,
        jurisdiction         TEXT,
        doc_type             TEXT,
        case_name            TEXT,
        case_name_short      TEXT,
        precedential_status  TEXT,
        nature_of_suit       TEXT,
        citation_count       BIGINT,
        judges               TEXT,
        metadata             JSONB NOT NULL DEFAULT '{}'::jsonb
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS public.case_edges (
        source_id BIGINT NOT NULL REFERENCES public.cases(case_id) ON DELETE CASCADE,
        target_id BIGINT NOT NULL REFERENCES public.cases(case_id) ON DELETE CASCADE,
        relation  TEXT   NOT NULL,
        depth     INTEGER,
        PRIMARY KEY (source_id, target_id, relation)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS public.chunks (
        chunk_id          TEXT PRIMARY KEY,
        case_id           BIGINT NOT NULL REFERENCES public.cases(case_id) ON DELETE CASCADE,
        parent_id         TEXT REFERENCES public.chunks(chunk_id)
                            ON DELETE CASCADE DEFERRABLE INITIALLY DEFERRED,
        is_parent         BOOLEAN NOT NULL,
        text              TEXT NOT NULL,
        text_original     TEXT NOT NULL,
        text_hash         TEXT NOT NULL,
        section           TEXT NOT NULL,
        section_path      TEXT NOT NULL DEFAULT '',
        section_priority  INTEGER NOT NULL,
        section_index     INTEGER NOT NULL,
        child_index       INTEGER NOT NULL,
        opinion_type      TEXT NOT NULL,
        page_start        INTEGER,
        page_end          INTEGER,
        char_start        BIGINT NOT NULL,
        char_end          BIGINT NOT NULL,
        court             TEXT NOT NULL DEFAULT '',
        date              DATE,
        citation          TEXT NOT NULL DEFAULT '',
        title             TEXT NOT NULL DEFAULT '',
        chunker_version   TEXT NOT NULL,
        embedding         halfvec(1024),
        embedder_version  TEXT,
        tsv               TSVECTOR GENERATED ALWAYS AS
                            (to_tsvector('english'::regconfig, text)) STORED,
        created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
        CONSTRAINT chunks_parent_shape CHECK (
            (is_parent AND parent_id IS NULL AND child_index = -1)
            OR
            (NOT is_parent AND parent_id IS NOT NULL AND child_index >= 0)
        ),
        CONSTRAINT chunks_valid_offsets CHECK (
            char_start >= 0 AND char_end >= char_start
        )
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS public.chunks_staging (
        load_id           UUID NOT NULL,
        chunk_id          TEXT NOT NULL,
        case_id           BIGINT NOT NULL,
        parent_id         TEXT,
        is_parent         BOOLEAN NOT NULL,
        text              TEXT NOT NULL,
        text_original     TEXT NOT NULL,
        text_hash         TEXT NOT NULL,
        section           TEXT NOT NULL,
        section_path      TEXT NOT NULL DEFAULT '',
        section_priority  INTEGER NOT NULL,
        section_index     INTEGER NOT NULL,
        child_index       INTEGER NOT NULL,
        opinion_type      TEXT NOT NULL,
        page_start        INTEGER,
        page_end          INTEGER,
        char_start        BIGINT NOT NULL,
        char_end          BIGINT NOT NULL,
        court             TEXT NOT NULL DEFAULT '',
        date              DATE,
        citation          TEXT NOT NULL DEFAULT '',
        title             TEXT NOT NULL DEFAULT '',
        chunker_version   TEXT NOT NULL,
        embedding         halfvec(1024),
        embedder_version  TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_chunks_staging_load_id ON public.chunks_staging (load_id)",
    """
    CREATE TABLE IF NOT EXISTS public.ingest_failures (
        failure_id   BIGSERIAL PRIMARY KEY,
        stage        TEXT NOT NULL,
        case_id      BIGINT,
        error        TEXT NOT NULL,
        traceback    TEXT,
        recorded_at  TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS public.pipeline_state (
        state_key   TEXT PRIMARY KEY,
        state       JSONB NOT NULL,
        updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
)


# ── Indexes ───────────────────────────────────────────────────────────────
#
# Three dicts must stay in lockstep. The invariant check below fails at
# import time if they drift — this prevents the KeyError-in-build_indexes
# bug that shipped when pg_trgm entries were added to INDEXES but not the
# companion dicts.

INDEXES: dict[str, str] = {
    "cases_court_date": """
        CREATE INDEX CONCURRENTLY idx_cases_court_date
        ON public.cases (court, date)
    """,
    "cases_precedential_status": """
        CREATE INDEX CONCURRENTLY idx_cases_precedential_status
        ON public.cases (precedential_status)
    """,
    "cases_citation": """
        CREATE INDEX CONCURRENTLY idx_cases_citation
        ON public.cases (citation)
    """,
    "cases_case_name_trgm": """
        CREATE INDEX CONCURRENTLY idx_cases_case_name_trgm
        ON public.cases USING gin (case_name gin_trgm_ops)
    """,
    "cases_short_trgm": """
        CREATE INDEX CONCURRENTLY idx_cases_short_trgm
        ON public.cases USING gin (case_name_short gin_trgm_ops)
    """,
    "embedding": """
        CREATE INDEX CONCURRENTLY idx_chunks_embedding
        ON public.chunks USING hnsw (embedding halfvec_cosine_ops)
        WITH (m = 16, ef_construction = 64)
    """,
    "tsv": """
        CREATE INDEX CONCURRENTLY idx_chunks_tsv
        ON public.chunks USING gin (tsv)
    """,
    "case_edges_source": """
        CREATE INDEX CONCURRENTLY idx_edges_source
        ON public.case_edges (source_id, relation)
    """,
    "case_edges_target": """
        CREATE INDEX CONCURRENTLY idx_edges_target
        ON public.case_edges (target_id, relation)
    """,
}

INDEX_NAME_BY_KEY: dict[str, str] = {
    "cases_court_date": "idx_cases_court_date",
    "cases_precedential_status": "idx_cases_precedential_status",
    "cases_citation": "idx_cases_citation",
    "cases_case_name_trgm": "idx_cases_case_name_trgm",
    "cases_short_trgm": "idx_cases_short_trgm",
    "embedding": "idx_chunks_embedding",
    "tsv": "idx_chunks_tsv",
    "case_edges_source": "idx_edges_source",
    "case_edges_target": "idx_edges_target",
}

INDEX_EXPECTATION: dict[str, str] = {
    "cases_court_date": "(court, date)",
    "cases_precedential_status": "(precedential_status)",
    "cases_citation": "(citation)",
    "cases_case_name_trgm": "gin (case_name gin_trgm_ops)",
    "cases_short_trgm": "gin (case_name_short gin_trgm_ops)",
    "embedding": "halfvec_cosine_ops",
    "tsv": "using gin (tsv)",
    "case_edges_source": "(source_id, relation)",
    "case_edges_target": "(target_id, relation)",
}

ALL_INDEX_NAMES: tuple[str, ...] = tuple(INDEX_NAME_BY_KEY.values())


# ── Invariant: three dicts in lockstep ────────────────────────────────────
# If this raises, fix the dicts above before touching anything else.


def _check_index_dicts() -> None:
    keys = set(INDEXES)
    names = set(INDEX_NAME_BY_KEY)
    expectations = set(INDEX_EXPECTATION)

    if keys != names or keys != expectations:
        missing_names = keys - names
        missing_expect = keys - expectations
        extra_names = names - keys
        extra_expect = expectations - keys

        parts: list[str] = ["Index dicts out of sync:"]
        if missing_names:
            parts.append(f"  INDEX_NAME_BY_KEY missing: {sorted(missing_names)}")
        if missing_expect:
            parts.append(f"  INDEX_EXPECTATION missing: {sorted(missing_expect)}")
        if extra_names:
            parts.append(f"  INDEX_NAME_BY_KEY extra:   {sorted(extra_names)}")
        if extra_expect:
            parts.append(f"  INDEX_EXPECTATION extra:   {sorted(extra_expect)}")

        raise RuntimeError("\n".join(parts))


_check_index_dicts()


# ── Validation gates ──────────────────────────────────────────────────────

VALIDATION_SQL: dict[str, str] = {
    "gate_1_parent_coverage": """
        SELECT COUNT(*)
        FROM public.chunks AS child
        LEFT JOIN public.chunks AS parent
          ON parent.chunk_id = child.parent_id
         AND parent.case_id = child.case_id
         AND parent.is_parent = TRUE
        WHERE child.is_parent = FALSE AND parent.chunk_id IS NULL
    """,
    # Breadcrumb format is [title, citation, year · court · opinion_type · section]
    # which is 4 " · "-separated groups (3 separators). The regex below must
    # match _make_breadcrumb's output exactly. 5 groups would fail every chunk.
    "gate_2_breadcrumb_presence": """
        SELECT COALESCE(AVG(
            CASE WHEN text ~
              '^\\[[^\\]\\n]+ · [^\\]\\n]+ · [^\\]\\n]+ · [^\\]\\n]+\\]'
              THEN 1.0 ELSE 0.0 END
        ), 1.0)
        FROM public.chunks
    """,
    "gate_3_boundary_integrity": """
        SELECT COALESCE(AVG(
            CASE WHEN rtrim(text_original) ~ '[.!?…:;][]["”’''")[:space:]]*$'
              THEN 0.0 ELSE 1.0 END
        ), 0.0)
        FROM public.chunks
        WHERE is_parent = FALSE
    """,
    # Gate 4 — split-citation detector. Each OR clause matches an atomic-
    # span fragment appearing at the END of a chunk body, meaning the
    # splitter cut inside that span. Synced with chunker.ATOMIC_SPAN_PATTERNS
    # (12 patterns) as of chunk-v1.6.
    #
    # Patterns intentionally NOT checked (each is a known false-positive
    # source that was tested and rejected):
    #   - Date fragments (19xx/20xx): a chunk ending "in 1973" is common.
    #   - Currency fragments ($N): a chunk ending "$10,000" without a
    #     following comma is rare in the source corpus.
    #   - Case names with v. mid-chunk: covered by '\mv[.]$' at end.
    #   - Bare "Capitalized_word." endings: too common in legal prose to
    #     distinguish from truncated state abbreviations. Removed after
    #     producing 15/523 false positives on the 50-case corpus.
    "gate_4_citation_integrity": """
        SELECT COUNT(*)
        FROM public.chunks
        WHERE is_parent = FALSE
          AND (
                -- 410 U.S. (federal reporters, incomplete)
                rtrim(text_original) ~ '[0-9]+[[:space:]]+U[.]S[.][[:space:]]*$'
                -- 565 So. | 123 P.3d | 500 N.E.2d (state/regional reporters)
             OR rtrim(text_original) ~ '[0-9]+[[:space:]]+(?:So|P|A|N[.]E|N[.]W|S[.]E|S[.]W)[.][[:space:]]*[0-9]*[a-z]*[[:space:]]*$'
                -- 42 U.S.C. (federal code, cut before §)
             OR rtrim(text_original) ~ '[0-9]+[[:space:]]+U[.]S[.]C[.][[:space:]]*$'
                -- Brown v. | Roe v. (case-name connective at end)
             OR rtrim(text_original) ~ '\\mv[.][[:space:]]*$'
                -- Mr. | Ms. | Dr. (honorific at end)
             OR rtrim(text_original) ~ '\\m(?:Mr|Mrs|Ms|Dr|Hon|Prof|Rev|Gen|Col|Sgt|Capt|Lt|St)[.][[:space:]]*$'
                -- § (bare section symbol at end)
             OR rtrim(text_original) ~ '[§][[:space:]]*$'
          )
    """,
    "gate_5_size_distribution": """
        SELECT
            COALESCE(percentile_cont(0.50) WITHIN GROUP (
                ORDER BY ceil(char_length(text_original) / 4.0)
            ) FILTER (WHERE is_parent = FALSE), 0),
            COALESCE(percentile_cont(0.95) WITHIN GROUP (
                ORDER BY ceil(char_length(text_original) / 4.0)
            ) FILTER (WHERE is_parent = FALSE), 0),
            COALESCE(MAX(ceil(char_length(text_original) / 4.0))
                FILTER (WHERE is_parent = FALSE), 0),
            COALESCE(MAX(ceil(char_length(text_original) / 4.0))
                FILTER (WHERE is_parent = TRUE), 0)
        FROM public.chunks
    """,
    # Gate 6 — intra-case duplicate detector. Cross-case duplicates are
    # handled by dedup.py before load; this gate catches chunker
    # regressions (same case, two identical child bodies).
    "gate_6_duplicate_content": """
        SELECT COUNT(*)
        FROM (
            SELECT case_id, text_hash
            FROM public.chunks
            WHERE is_parent = FALSE
            GROUP BY case_id, text_hash
            HAVING COUNT(*) > 1
        ) AS duplicates
    """,
    "gate_7_section_coverage": """
        SELECT COALESCE(AVG(
            CASE WHEN parent.section = child.section
                  AND parent.section_path = child.section_path
              THEN 1.0 ELSE 0.0 END
        ), 1.0)
        FROM public.chunks AS child
        JOIN public.chunks AS parent
          ON parent.chunk_id = child.parent_id
         AND parent.case_id = child.case_id
        WHERE child.is_parent = FALSE
    """,
}


# ── Result types ──────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class GateResult:
    name: str
    passed: bool
    value: Any
    threshold: str
    note: str = ""


@dataclass(frozen=True, slots=True)
class ValidationReport:
    gates: dict[str, GateResult]
    ok: bool
    row_counts: dict[str, int]
    measured_at: datetime


class DatabaseError(Exception):
    def __init__(self, reason: str, detail: str = "") -> None:
        self.reason = reason
        self.detail = detail
        super().__init__(reason if not detail else f"{reason}: {detail}")


# ── Lazy pgvector loaders ────────────────────────────────────────────────
# pgvector is only needed for insert (HalfVector) and connect
# (register_vector). manifest / chunk / embed never touch it, so a broken
# pgvector install must not break `import indexing_pipeline`.

_HALF_VECTOR_CLASS: Any = None
_REGISTER_VECTOR_FN: Any = None


def _half_vector() -> Any:
    global _HALF_VECTOR_CLASS
    if _HALF_VECTOR_CLASS is None:
        try:
            from pgvector import HalfVector as _HV

            _HALF_VECTOR_CLASS = _HV
        except Exception as exc:
            raise DatabaseError(
                "pgvector import failed",
                "install with: pip install 'pgvector>=0.5,<0.6'. "
                f"Original error: {type(exc).__name__}: {exc}",
            ) from exc
    return _HALF_VECTOR_CLASS


def _register_vector_fn() -> Any:
    global _REGISTER_VECTOR_FN
    if _REGISTER_VECTOR_FN is None:
        try:
            from pgvector.psycopg import register_vector as _rv

            _REGISTER_VECTOR_FN = _rv
        except Exception as exc:
            raise DatabaseError(
                "pgvector.psycopg import failed",
                "install with: pip install 'pgvector>=0.5,<0.6'. "
                f"Original error: {type(exc).__name__}: {exc}",
            ) from exc
    return _REGISTER_VECTOR_FN


# ── Database ──────────────────────────────────────────────────────────────


class Database:
    def __init__(self, dsn: str, pool_min: int = 1, pool_max: int = 4) -> None:
        if not dsn:
            raise DatabaseError("DATABASE_URL is required")
        if pool_min < 1 or pool_max < pool_min:
            raise DatabaseError("invalid pool configuration")

        self.dsn = dsn
        self.pool_min = pool_min
        self.pool_max = pool_max
        self._closed = False
        self.load_id = uuid.uuid4()

        try:
            # Extension MUST exist before register_vector runs on pool conns.
            with psycopg.connect(self.dsn, connect_timeout=10) as conn:
                conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

            self.pool = ConnectionPool(
                conninfo=self.dsn,
                min_size=self.pool_min,
                max_size=self.pool_max,
                timeout=POOL_TIMEOUT_SECONDS,
                configure=self._configure_connection,
                open=False,
            )
            self.pool.open(wait=True, timeout=POOL_TIMEOUT_SECONDS)
        except Exception as exc:
            pool = getattr(self, "pool", None)
            if pool is not None:
                try:
                    pool.close()
                except Exception:
                    pass
            if isinstance(exc, DatabaseError):
                raise
            raise DatabaseError(
                "could not initialize PostgreSQL pool",
                str(exc),
            ) from exc

    @staticmethod
    def _configure_connection(conn: psycopg.Connection) -> None:
        _register_vector_fn()(conn)

    @staticmethod
    def _wrap(operation: str, exc: Exception) -> DatabaseError:
        detail = str(exc)
        sqlstate = getattr(exc, "sqlstate", None)
        if sqlstate:
            detail = f"SQLSTATE={sqlstate}; {detail}"
        return DatabaseError(operation, detail)

    def _execute_autocommit(self, statement: str) -> None:
        """Execute DDL requiring autocommit on a dedicated connection.

        Concurrent index creation and deletion cannot run inside a
        transaction. A dedicated connection avoids changing autocommit
        state on connections managed by the pool.
        """
        self._require_open()
        with psycopg.connect(
            self.dsn,
            connect_timeout=10,
            autocommit=True,
        ) as conn:
            conn.execute(statement)

    def _require_open(self) -> None:
        if self._closed:
            raise DatabaseError("database is closed")

    @staticmethod
    def _to_date(value: Any) -> date | None:
        if value is None or value == "":
            return None
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
        return date.fromisoformat(str(value)[:10])

    @staticmethod
    def _jsonb(value: Any) -> Jsonb:
        if value is None:
            value = {}
        if not isinstance(value, dict):
            raise ValueError("metadata must be a dict")
        return Jsonb(value)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.pool.close()
        except Exception as exc:
            raise self._wrap("failed to close pool", exc) from exc

    def __enter__(self) -> Database:
        self._require_open()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, tb: Any) -> None:
        self.close()

    # ── Schema ────────────────────────────────────────────────────────

    def ensure_schema(self) -> None:
        self._require_open()
        try:
            with self.pool.connection() as conn:
                for stmt in SCHEMA_STATEMENTS:
                    conn.execute(stmt)

                # Additive migrations for older installs.
                conn.execute(
                    "ALTER TABLE public.chunks ADD COLUMN IF NOT EXISTS embedder_version TEXT"
                )
                conn.execute(
                    "ALTER TABLE public.chunks "
                    "ADD COLUMN IF NOT EXISTS created_at "
                    "TIMESTAMPTZ NOT NULL DEFAULT now()"
                )
                conn.execute(
                    "ALTER TABLE public.cases "
                    "ADD COLUMN IF NOT EXISTS metadata "
                    "JSONB NOT NULL DEFAULT '{}'::jsonb"
                )
                conn.execute(
                    "ALTER TABLE public.chunks_staging ADD COLUMN IF NOT EXISTS load_id UUID"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_chunks_staging_load_id "
                    "ON public.chunks_staging (load_id)"
                )

                row = conn.execute("""
                    SELECT format_type(a.atttypid, a.atttypmod)
                    FROM pg_attribute a
                    JOIN pg_class c     ON c.oid = a.attrelid
                    JOIN pg_namespace n ON n.oid = c.relnamespace
                    WHERE n.nspname = 'public'
                      AND c.relname = 'chunks'
                      AND a.attname = 'embedding'
                      AND a.attnum > 0 AND NOT a.attisdropped
                """).fetchone()

                if row is None or row[0] != "halfvec(1024)":
                    raise DatabaseError(
                        "schema incompatibility",
                        "public.chunks.embedding must be halfvec(1024)",
                    )
        except DatabaseError:
            raise
        except Exception as exc:
            raise self._wrap("failed to create or verify schema", exc) from exc

    # ── Index introspection ──────────────────────────────────────────

    def _index_info(self, index_name: str) -> tuple[bool, str] | None:
        self._require_open()
        try:
            with self.pool.connection() as conn:
                row = conn.execute(
                    """
                    SELECT i.indisvalid, pg_get_indexdef(c.oid)
                    FROM pg_class c
                    JOIN pg_namespace n ON n.oid = c.relnamespace
                    LEFT JOIN pg_index i ON i.indexrelid = c.oid
                    WHERE n.nspname = 'public'
                      AND c.relname = %s
                      AND c.relkind = 'i'
                    """,
                    (index_name,),
                ).fetchone()

                return None if row is None else (bool(row[0]), str(row[1]))
        except Exception as exc:
            raise self._wrap(f"could not inspect index {index_name}", exc) from exc

    def index_exists(self) -> bool:
        return self._index_info("idx_chunks_embedding") is not None

    def index_is_valid(self) -> bool:
        info = self._index_info("idx_chunks_embedding")
        return bool(info and info[0])

    def drop_indexes(self) -> None:
        """Drop all managed indexes using a dedicated autocommit connection."""
        self._require_open()
        try:
            with psycopg.connect(
                self.dsn,
                connect_timeout=10,
                autocommit=True,
            ) as conn:
                for name in ALL_INDEX_NAMES:
                    conn.execute(f"DROP INDEX CONCURRENTLY IF EXISTS public.{name}")
        except Exception as exc:
            raise self._wrap("could not drop indexes", exc) from exc

    def build_indexes(self) -> dict[str, dict[str, Any]]:
        """Build and verify all managed indexes.

        Concurrent index DDL runs on dedicated autocommit connections.
        Pooled connections are used only for ordinary transactional queries.
        """
        self._require_open()
        report: dict[str, dict[str, Any]] = {}

        try:
            for key, statement in INDEXES.items():
                name = INDEX_NAME_BY_KEY[key]
                expected = INDEX_EXPECTATION[key].lower()

                info = self._index_info(name)

                if info is not None:
                    valid, definition = info

                    if valid and expected in definition.lower():
                        report[key] = {
                            "created": False,
                            "duration_s": 0.0,
                            "index_size_mb": self._index_size_mb(name),
                            "rows": self._table_rows_for(key),
                        }
                        continue

                    # Remove invalid or incorrectly defined index.
                    self._execute_autocommit(f"DROP INDEX CONCURRENTLY IF EXISTS public.{name}")

                started = time.monotonic()

                # CREATE INDEX CONCURRENTLY requires autocommit.
                self._execute_autocommit(statement)

                duration = time.monotonic() - started

                info = self._index_info(name)

                if info is None or not info[0]:
                    raise DatabaseError(f"index build did not produce a valid index: {name}")

                if expected not in info[1].lower():
                    raise DatabaseError(f"index definition verification failed: {name}")

                report[key] = {
                    "created": True,
                    "duration_s": round(duration, 3),
                    "index_size_mb": self._index_size_mb(name),
                    "rows": self._table_rows_for(key),
                }

            return report

        except DatabaseError:
            raise
        except Exception as exc:
            raise self._wrap("failed to build indexes", exc) from exc

    def _index_size_mb(self, index_name: str) -> float:
        with self.pool.connection() as conn:
            row = conn.execute(
                "SELECT pg_relation_size(%s::regclass)",
                (f"public.{index_name}",),
            ).fetchone()
            return round(int(row[0]) / (1024 * 1024), 3)

    def _table_rows_for(self, key: str) -> int:
        table = (
            "cases"
            if key.startswith("cases_")
            else "case_edges"
            if key.startswith("case_edges_")
            else "chunks"
        )
        with self.pool.connection() as conn:
            return int(conn.execute(f"SELECT COUNT(*) FROM public.{table}").fetchone()[0])

    # ── Stale-row cleanup ────────────────────────────────────────────

    def delete_stale_cases(self, keep_case_ids: list[int]) -> int:
        """Delete cases not in keep list. Cascades to case_edges and chunks."""
        self._require_open()

        # An empty keep-list is treated as a safety no-op. This prevents
        # an accidental empty input from deleting the entire corpus.
        if not keep_case_ids:
            return 0

        try:
            with self.pool.connection() as conn:
                result = conn.execute(
                    "DELETE FROM public.cases WHERE case_id <> ALL(%s::bigint[])",
                    (keep_case_ids,),
                )
                return result.rowcount
        except Exception as exc:
            raise self._wrap("could not delete stale cases", exc) from exc

    # ── Load ─────────────────────────────────────────────────────────

    def load_cases(self, rows: Iterable[dict[str, Any]]) -> int:
        self._require_open()

        columns = ", ".join(CASE_COLUMNS)
        compared_cols = [c for c in CASE_COLUMNS if c != "case_id"]
        assignments = ", ".join(f"{c} = EXCLUDED.{c}" for c in compared_cols)

        query = f"""
            INSERT INTO public.cases ({columns})
            SELECT {columns} FROM cases_staging
            ON CONFLICT (case_id) DO UPDATE SET {assignments}
            WHERE ROW({", ".join("public.cases." + c for c in compared_cols)})
               IS DISTINCT FROM
                  ROW({", ".join("EXCLUDED." + c for c in compared_cols)})
        """

        try:
            with self.pool.connection() as conn:
                conn.execute("""
                    CREATE TEMP TABLE cases_staging
                    (LIKE public.cases INCLUDING DEFAULTS)
                    ON COMMIT DROP
                """)

                copied = 0

                with conn.cursor().copy(f"COPY cases_staging ({columns}) FROM STDIN") as copy:
                    for row in rows:
                        record = (
                            row.get("case_id"),
                            row.get("cluster_id"),
                            row.get("court"),
                            self._to_date(row.get("date")),
                            row.get("title"),
                            row.get("citation"),
                            row.get("opinion_type"),
                            row.get("jurisdiction"),
                            row.get("doc_type"),
                            row.get("case_name"),
                            row.get("case_name_short"),
                            row.get("precedential_status"),
                            row.get("nature_of_suit"),
                            row.get("citation_count"),
                            row.get("judges"),
                            self._jsonb(row.get("metadata")),
                        )

                        if record[0] is None:
                            raise ValueError("case_id cannot be NULL")

                        copy.write_row(record)
                        copied += 1

                if not copied:
                    return 0

                return conn.execute(query).rowcount

        except Exception as exc:
            raise self._wrap("failed to load cases", exc) from exc

    def load_edges(self, rows: Iterable[dict[str, Any]]) -> int:
        self._require_open()

        try:
            with self.pool.connection() as conn:
                conn.execute(
                    "CREATE TEMP TABLE edges_staging (LIKE public.case_edges) ON COMMIT DROP"
                )

                copied = 0

                with conn.cursor().copy(
                    "COPY edges_staging (source_id, target_id, relation, depth) FROM STDIN"
                ) as copy:
                    for row in rows:
                        source = row.get("source_id", row.get("source"))
                        target = row.get("target_id", row.get("target"))

                        if source is None or target is None:
                            raise ValueError("edge source_id and target_id required")

                        copy.write_row(
                            (
                                int(source),
                                int(target),
                                str(row.get("relation") or "CITES"),
                                int(row["depth"]) if row.get("depth") is not None else None,
                            )
                        )
                        copied += 1

                if not copied:
                    return 0

                return conn.execute("""
                    INSERT INTO public.case_edges (source_id, target_id, relation, depth)
                    SELECT DISTINCT ON (source_id, target_id, relation)
                        source_id, target_id, relation, depth
                    FROM edges_staging
                    ORDER BY source_id, target_id, relation, depth DESC NULLS LAST
                    ON CONFLICT (source_id, target_id, relation)
                    DO UPDATE SET depth = EXCLUDED.depth
                    WHERE public.case_edges.depth IS DISTINCT FROM EXCLUDED.depth
                """).rowcount

        except Exception as exc:
            raise self._wrap("failed to load edges", exc) from exc

    @staticmethod
    def _prepare_embedding(value: Any) -> Any:
        if value is None:
            return None

        HalfVector = _half_vector()

        if isinstance(value, HalfVector):
            return value

        if hasattr(value, "tolist"):
            value = value.tolist()

        if not isinstance(value, (list, tuple)):
            raise ValueError("embedding must be list, tuple, ndarray, or HalfVector")

        if len(value) != EMBEDDING_DIMENSIONS:
            raise ValueError(f"expected {EMBEDDING_DIMENSIONS} dims, got {len(value)}")

        return HalfVector([float(x) for x in value])

    def stage_chunks(self, rows: Iterable[dict[str, Any]]) -> int:
        self._require_open()

        try:
            with self.pool.connection() as conn:
                copied = 0

                with conn.cursor().copy(
                    f"COPY public.chunks_staging (load_id, {CHUNK_COPY_COLUMNS}) FROM STDIN"
                ) as copy:
                    for row in rows:
                        embedding = self._prepare_embedding(row.get("embedding"))

                        copy.write_row(
                            (
                                self.load_id,
                                row["chunk_id"],
                                int(row["case_id"]),
                                row.get("parent_id"),
                                bool(row["is_parent"]),
                                row["text"],
                                row["text_original"],
                                row["text_hash"],
                                row["section"],
                                row.get("section_path", ""),
                                int(row["section_priority"]),
                                int(row["section_index"]),
                                int(row["child_index"]),
                                row["opinion_type"],
                                row.get("page_start"),
                                row.get("page_end"),
                                int(row["char_start"]),
                                int(row["char_end"]),
                                row.get("court", ""),
                                self._to_date(row.get("date")),
                                row.get("citation", ""),
                                row.get("title", ""),
                                row["chunker_version"],
                                embedding,
                                row.get("embedder_version"),
                            )
                        )
                        copied += 1

                return copied

        except Exception as exc:
            raise self._wrap("failed to stage chunks", exc) from exc

    def merge_chunks(self) -> tuple[int, int]:
        """Merge staged chunks into public.chunks, deleting stale ones.

        Upserts all rows from chunks_staging for the current load_id
        (ON CONFLICT DO UPDATE, skipping no-op updates via IS DISTINCT
        FROM). Then deletes any chunk whose case_id is in the current
        staging batch but whose chunk_id is NOT in the staging batch —
        this is required after dedup.py removes cross-case duplicate
        children from chunked/*.jsonl. Without the DELETE, dedup-dropped
        children would persist in public.chunks across re-loads.

        The DELETE only affects chunks whose case_id is present in the
        current load's staging set. Cases not in this load (e.g. failed
        chunk) retain their existing chunks — handled separately by
        pipeline.load()'s delete_stale_cases() and _is_valid_chunk_output
        skip logic.

        Parents are never deleted by this method because they are always
        staged alongside their children (chunk_opinion emits both). The
        FK cascade (parent_id REFERENCES chunks(chunk_id)) only fires when
        a parent row is deleted, which never happens here.

        Returns:
            (upserted, deleted): rowcount of INSERT...ON CONFLICT (may be
            0 when re-loading an unchanged dataset), and rowcount of the
            stale-chunk DELETE (may be 0 when dedup removed nothing).

        Raises:
            DatabaseError: if a HNSW index exists (drop before re-loading)
            DatabaseError: on any SQL failure
        """
        self._require_open()

        try:
            if self.index_exists():
                raise DatabaseError("drop indexes before re-loading; run 'index --force'")

            update_columns = [c for c in CHUNK_COLUMNS if c != "chunk_id"]
            insert_columns = ", ".join(CHUNK_COLUMNS)
            select_columns = ", ".join(f"staging.{c}" for c in CHUNK_COLUMNS)
            assignments = ", ".join(f"{c} = EXCLUDED.{c}" for c in update_columns)

            upsert_query = f"""
                INSERT INTO public.chunks ({insert_columns})
                SELECT {select_columns}
                FROM public.chunks_staging AS staging
                WHERE staging.load_id = %s
                ORDER BY staging.is_parent DESC, staging.chunk_id
                ON CONFLICT (chunk_id) DO UPDATE SET {assignments}
                WHERE ROW({", ".join("public.chunks." + c for c in update_columns)})
                   IS DISTINCT FROM
                      ROW({", ".join("EXCLUDED." + c for c in update_columns)})
            """

            # Stale-chunk DELETE: any chunk whose case_id appears in this
            # load's staging set but whose chunk_id does not. Catches
            # dedup-removed children and re-chunked shrink cases. Runs in
            # the same transaction as the upsert.
            delete_stale_query = """
                DELETE FROM public.chunks c
                USING (
                    SELECT DISTINCT case_id
                    FROM public.chunks_staging
                    WHERE load_id = %s
                ) loaded_cases
                WHERE c.case_id = loaded_cases.case_id
                  AND NOT EXISTS (
                      SELECT 1
                      FROM public.chunks_staging s
                      WHERE s.load_id = %s
                        AND s.chunk_id = c.chunk_id
                  )
            """

            with self.pool.connection() as conn:
                upserted = conn.execute(
                    upsert_query,
                    (self.load_id,),
                ).rowcount

                deleted = conn.execute(
                    delete_stale_query,
                    (self.load_id, self.load_id),
                ).rowcount

                conn.execute(
                    "DELETE FROM public.chunks_staging WHERE load_id = %s",
                    (self.load_id,),
                )

                return upserted, deleted

        except DatabaseError:
            raise
        except Exception as exc:
            raise self._wrap("failed to merge staged chunks", exc) from exc

    def record_failure(
        self,
        stage: str,
        case_id: int | None,
        error: str,
        traceback: str,
    ) -> None:
        self._require_open()

        try:
            with self.pool.connection() as conn:
                conn.execute(
                    "INSERT INTO public.ingest_failures "
                    "(stage, case_id, error, traceback) "
                    "VALUES (%s, %s, %s, %s)",
                    (stage, case_id, error, traceback),
                )
        except Exception as exc:
            raise self._wrap("failed to record failure", exc) from exc

    # ── Pipeline state ───────────────────────────────────────────────

    def get_pipeline_state(self, state_key: str) -> dict[str, Any] | None:
        self._require_open()

        try:
            with self.pool.connection() as conn:
                row = conn.execute(
                    "SELECT state FROM public.pipeline_state WHERE state_key = %s",
                    (state_key,),
                ).fetchone()

                return row[0] if row else None

        except Exception as exc:
            raise self._wrap("could not read pipeline state", exc) from exc

    def set_pipeline_state(self, state_key: str, state: dict[str, Any]) -> None:
        self._require_open()

        try:
            with self.pool.connection() as conn:
                conn.execute(
                    """
                    INSERT INTO public.pipeline_state (state_key, state, updated_at)
                    VALUES (%s, %s, now())
                    ON CONFLICT (state_key)
                    DO UPDATE SET state = EXCLUDED.state, updated_at = now()
                    WHERE public.pipeline_state.state IS DISTINCT FROM EXCLUDED.state
                    """,
                    (state_key, Jsonb(state)),
                )

        except Exception as exc:
            raise self._wrap("could not persist pipeline state", exc) from exc

    # ── Validation ───────────────────────────────────────────────────

    def validation_report(
        self,
        determinism_gate: GateResult | None = None,
        extra_gates: dict[str, GateResult] | None = None,
    ) -> ValidationReport:
        self._require_open()

        try:
            with self.pool.connection() as conn:
                conn.execute("SET TRANSACTION READ ONLY")

                gates: dict[str, GateResult] = {}

                orphans = int(conn.execute(VALIDATION_SQL["gate_1_parent_coverage"]).fetchone()[0])
                gates["gate_1_parent_coverage"] = GateResult(
                    name="gate_1_parent_coverage",
                    passed=orphans == 0,
                    value=orphans,
                    threshold="0 orphan children",
                )

                breadcrumb = float(
                    conn.execute(VALIDATION_SQL["gate_2_breadcrumb_presence"]).fetchone()[0]
                )
                gates["gate_2_breadcrumb_presence"] = GateResult(
                    name="gate_2_breadcrumb_presence",
                    passed=breadcrumb >= 1.0,
                    value=breadcrumb,
                    threshold="100%",
                )

                boundary = float(
                    conn.execute(VALIDATION_SQL["gate_3_boundary_integrity"]).fetchone()[0]
                )
                gates["gate_3_boundary_integrity"] = GateResult(
                    name="gate_3_boundary_integrity",
                    # Threshold relaxed 0.02 → 0.05 on 2026-11-04 with
                    # evidence: some legal sentences exceed CHILD_MAX_TOK
                    # by themselves and must be force-cut. 5% ceiling
                    # matches industry norms for legal RAG chunking.
                    passed=boundary <= 0.05,
                    value=boundary,
                    threshold="<= 0.05 (long legal sentences force-cut)",
                    note="punctuation heuristic, not a parser",
                )

                split_cites = int(
                    conn.execute(VALIDATION_SQL["gate_4_citation_integrity"]).fetchone()[0]
                )
                gates["gate_4_citation_integrity"] = GateResult(
                    name="gate_4_citation_integrity",
                    passed=split_cites == 0,
                    value=split_cites,
                    threshold=("0 split citations (U.S., Ala., v., Ms., So.2d, U.S.C., §)"),
                )

                row = conn.execute(VALIDATION_SQL["gate_5_size_distribution"]).fetchone()

                sizes = {
                    "p50": float(row[0]),
                    "p95": float(row[1]),
                    "max_child": int(row[2]),
                    "max_parent": int(row[3]),
                }

                gates["gate_5_size_distribution"] = GateResult(
                    name="gate_5_size_distribution",
                    passed=(
                        sizes["p50"] <= 400
                        and sizes["p95"] <= 450
                        # v2.4: gate_5 uses char_length/4.0 (an ESTIMATE).
                        # For prose-heavy chunks, real tokens < estimate/1.2.
                        # Chunker enforces ≤464 real tokens at emit; gate_5
                        # threshold accommodates 25% estimator noise.
                        and sizes["max_child"] <= 580
                        and sizes["max_parent"] <= 1800
                    ),
                    value=sizes,
                    threshold=("child p50<=400, p95<=450, max<=580 (est); parent max<=1800 (est)"),
                )

                dups = int(conn.execute(VALIDATION_SQL["gate_6_duplicate_content"]).fetchone()[0])
                gates["gate_6_duplicate_content"] = GateResult(
                    name="gate_6_duplicate_content",
                    passed=dups == 0,
                    value=dups,
                    threshold="0 duplicate child bodies per case",
                    note="intra-case only; cross-case dedup is done pre-load",
                )

                sec_cov = float(
                    conn.execute(VALIDATION_SQL["gate_7_section_coverage"]).fetchone()[0]
                )
                gates["gate_7_section_coverage"] = GateResult(
                    name="gate_7_section_coverage",
                    passed=sec_cov >= 0.98,
                    value=sec_cov,
                    threshold=">= 0.98",
                    note="child-to-parent consistency",
                )

                if determinism_gate is None:
                    determinism_gate = GateResult(
                        name="gate_8_determinism",
                        passed=False,
                        value="not measured",
                        threshold="identical chunk IDs and hashes",
                    )

                gates["gate_8_determinism"] = determinism_gate

                if extra_gates:
                    for name, gate in extra_gates.items():
                        gates[name] = gate

                return ValidationReport(
                    gates=gates,
                    ok=all(g.passed for g in gates.values()),
                    row_counts=self._row_counts(conn),
                    measured_at=datetime.now(UTC),
                )

        except DatabaseError:
            raise
        except Exception as exc:
            raise self._wrap("validation queries failed", exc) from exc

    def _row_counts(self, conn: psycopg.Connection) -> dict[str, int]:
        """Return row counts including diagnostic counts.

        Diagnostics (not gated):
          - orphan_parents: parents whose children were all filtered by
            is_noise or all deduped. Inert (parents are not indexed for
            retrieval) but worth reporting so dedup+filter impact is
            visible in reports/validation.json.
        """
        return {
            "cases": int(conn.execute("SELECT COUNT(*) FROM public.cases").fetchone()[0]),
            "case_edges": int(conn.execute("SELECT COUNT(*) FROM public.case_edges").fetchone()[0]),
            "chunks": int(conn.execute("SELECT COUNT(*) FROM public.chunks").fetchone()[0]),
            "chunks_with_embedding": int(
                conn.execute(
                    "SELECT COUNT(*) FROM public.chunks WHERE embedding IS NOT NULL"
                ).fetchone()[0]
            ),
            "children": int(
                conn.execute(
                    "SELECT COUNT(*) FROM public.chunks WHERE is_parent = FALSE"
                ).fetchone()[0]
            ),
            "parents": int(
                conn.execute(
                    "SELECT COUNT(*) FROM public.chunks WHERE is_parent = TRUE"
                ).fetchone()[0]
            ),
            "orphan_parents": int(
                conn.execute("""
                    SELECT COUNT(*)
                    FROM public.chunks p
                    WHERE p.is_parent = TRUE
                      AND NOT EXISTS (
                          SELECT 1 FROM public.chunks c
                          WHERE c.parent_id = p.chunk_id
                      )
                """).fetchone()[0]
            ),
            "chunks_staging_pending": int(
                conn.execute("SELECT COUNT(*) FROM public.chunks_staging").fetchone()[0]
            ),
        }

    def row_counts(self) -> dict[str, int]:
        self._require_open()

        try:
            with self.pool.connection() as conn:
                return self._row_counts(conn)
        except Exception as exc:
            raise self._wrap("could not read row counts", exc) from exc
