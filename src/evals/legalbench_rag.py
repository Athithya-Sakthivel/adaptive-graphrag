"""Independent LegalBench-RAG loader and pgvector retriever."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import sys
import unicodedata
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

CHUNK_SIZE = 500
PROTOCOL_VERSION = "legalbench-naive-char-500-no-overlap-v1"
SUBSETS = ("contractnli", "cuad", "maud", "privacy_qa")


def normalise_path(value: str) -> str:
    return unicodedata.normalize("NFC", value.replace("\\", "/"))


def corpus_fingerprint(corpus_root: Path) -> str:
    if not corpus_root.is_dir():
        raise FileNotFoundError(corpus_root)
    digest = hashlib.sha256()
    digest.update(PROTOCOL_VERSION.encode())
    digest.update(b"\0")
    files = sorted(p for p in corpus_root.rglob("*.txt") if p.is_file())
    if not files:
        raise ValueError("No text files found in corpus")
    seen = set()
    for path in files:
        relative = normalise_path(path.relative_to(corpus_root).as_posix())
        if relative in seen:
            raise ValueError(f"Duplicate normalised path: {relative}")
        seen.add(relative)
        content = path.read_text(encoding="utf-8")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def document_map(corpus_root: Path) -> dict[str, Path]:
    documents: dict[str, Path] = {}
    for path in corpus_root.rglob("*.txt"):
        if not path.is_file():
            continue
        relative = normalise_path(path.relative_to(corpus_root).as_posix())
        if relative in documents:
            raise ValueError(f"Duplicate document path: {relative}")
        documents[relative] = path
    return documents


def iter_documents(corpus_root: Path) -> Iterator[tuple[str, str]]:
    documents = document_map(corpus_root)
    for relative in sorted(documents):
        yield relative, documents[relative].read_text(encoding="utf-8")


def iter_chunks(
    corpus_root: Path,
) -> Iterator[tuple[str, int, int, str, str]]:
    for doc, content in iter_documents(corpus_root):
        for start in range(0, len(content), CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, len(content))
            text = content[start:end]
            if not text:
                continue
            chunk_id = hashlib.sha256(f"{doc}\0{start}\0{end}".encode()).hexdigest()
            yield doc, start, end, chunk_id, text


def safe_benchmark_path(raw_path: str) -> str:
    from pathlib import PurePosixPath

    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("Invalid benchmark file path")
    raw = raw_path.replace("\\", "/")
    path = PurePosixPath(raw)
    if path.is_absolute() or ".." in path.parts or (path.parts and path.parts[0].endswith(":")):
        raise ValueError(f"Unsafe benchmark path: {raw_path}")
    return normalise_path(path.as_posix())


def load_benchmark_gold(
    *,
    corpus_root: Path,
    benchmarks_root: Path,
    subsets: Sequence[str],
    corpus_hash: str,
    dataset_revision: str,
) -> list[dict]:
    documents = document_map(corpus_root)
    text_cache = {r: p.read_text(encoding="utf-8") for r, p in documents.items()}
    gold = []
    for subset in subsets:
        benchmark_path = benchmarks_root / f"{subset}.json"
        if not benchmark_path.is_file():
            raise FileNotFoundError(benchmark_path)
        benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
        tests = benchmark.get("tests")
        if not isinstance(tests, list) or not tests:
            raise ValueError(f"{benchmark_path}: expected non-empty tests array")
        for index, test in enumerate(tests):
            query = test.get("query")
            snippets = test.get("snippets")
            if not isinstance(query, str) or not query.strip():
                raise ValueError(f"{subset}/{index}: invalid query")
            if not isinstance(snippets, list) or not snippets:
                raise ValueError(f"{subset}/{index}: missing snippets")
            spans = []
            for snippet in snippets:
                doc = safe_benchmark_path(snippet.get("file_path"))
                if doc not in text_cache:
                    raise FileNotFoundError(f"Missing LegalBench document: {doc}")
                raw_span = snippet.get("span")
                if isinstance(raw_span, (list, tuple)) and len(raw_span) == 2:
                    start, end = raw_span
                else:
                    start = snippet.get("start")
                    end = snippet.get("end")
                if (
                    isinstance(start, bool)
                    or isinstance(end, bool)
                    or not isinstance(start, int)
                    or not isinstance(end, int)
                    or start < 0
                    or end <= start
                    or end > len(text_cache[doc])
                ):
                    raise ValueError(f"{subset}/{index}: invalid span {doc}:{start}:{end}")
                supplied_answer = snippet.get("answer")
                if supplied_answer is not None and text_cache[doc][start:end] != supplied_answer:
                    raise ValueError(
                        f"{subset}/{index}: supplied answer does not "
                        "match source characters. Check offsets, "
                        "newlines, and BOM handling."
                    )
                spans.append({"doc": doc, "start": start, "end": end})
            gold.append(
                {
                    "id": f"legalbench/{subset}/{index:05d}",
                    "suite": "legalbench",
                    "query": query,
                    "query_class_expected": subset,
                    "answerable": True,
                    "gold_case_ids": [],
                    "gold_chunk_ids": [],
                    "gold_char_spans": spans,
                    "seed_case_id": None,
                    "tags": ["legalbench", subset],
                    "verified": True,
                    "corpus_hash": corpus_hash,
                    "verification_source": "external_legalbench_dataset",
                    "verification_method": (
                        "official character offsets checked against local corpus"
                    ),
                    "dataset_revision": dataset_revision,
                }
            )
    return gold


def write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def load_embedder(spec: str) -> Any:
    if ":" in spec:
        module_name, _ = spec.split(":", 1)
    else:
        module_name = spec
    module = importlib.import_module(module_name)
    for name in ("embed_documents", "embed_query"):
        if not callable(getattr(module, name, None)):
            raise TypeError(f"Embedder must expose {name}()")
    return module


def validate_vector(
    value: Sequence[float],
    expected_dimension: int | None = None,
) -> list[float]:
    vector = [float(x) for x in value]
    if not vector:
        raise ValueError("Empty embedding")
    if expected_dimension is not None and len(vector) != expected_dimension:
        raise ValueError("Embedding dimension changed")
    if not all(math.isfinite(x) for x in vector):
        raise ValueError("Non-finite embedding value")
    if sum(x * x for x in vector) <= 0:
        raise ValueError("Cosine search does not accept zero vectors")
    return vector


def vector_literal(vector: Sequence[float]) -> str:
    return "[" + ",".join(format(float(value), ".9g") for value in vector) + "]"


def batches(
    values: Iterable[Any],
    size: int,
) -> Iterator[list[Any]]:
    if size < 1:
        raise ValueError("batch size must be positive")
    current: list[Any] = []
    for value in values:
        current.append(value)
        if len(current) == size:
            yield current
            current = []
    if current:
        yield current


def build_postgres_index(
    *,
    corpus_root: Path,
    dsn: str,
    embedder: Any,
    embedding_model: str,
    corpus_hash: str,
    batch_size: int = 64,
    rebuild: bool = False,
) -> dict[str, Any]:
    import psycopg

    stream = iter_chunks(corpus_root)
    first_batch = next(batches(stream, batch_size), None)
    if not first_batch:
        raise ValueError("No chunks produced")

    def embed_batch(batch: list[Any]) -> tuple[list[list[float]], int]:
        raw = embedder.embed_documents(
            [i[4] for i in batch],
            model=embedding_model,
        )
        if len(raw) != len(batch):
            raise ValueError("Embedding count mismatch")
        vectors = [validate_vector(v) for v in raw]
        dimension = len(vectors[0])
        if any(len(v) != dimension for v in vectors):
            raise ValueError("Inconsistent dimensions")
        if dimension > 4000:
            raise ValueError("pgvector halfvec HNSW supports at most 4000 dimensions")
        return vectors, dimension

    first_vectors, dimension = embed_batch(first_batch)
    count = 0
    with psycopg.connect(dsn) as conn:
        with conn.transaction():
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            if rebuild:
                conn.execute("DROP TABLE IF EXISTS lbrag_chunks CASCADE")
                conn.execute("DROP TABLE IF EXISTS lbrag_meta CASCADE")
            exists = conn.execute("SELECT to_regclass('public.lbrag_chunks')").fetchone()[0]
            if exists:
                meta_exists = conn.execute("SELECT to_regclass('public.lbrag_meta')").fetchone()[0]
                if not meta_exists:
                    raise RuntimeError("Incomplete LegalBench index; use --rebuild")
                metadata = conn.execute(
                    "SELECT corpus_hash, embedding_model, "
                    "dimension, chunk_size, chunk_count "
                    "FROM lbrag_meta WHERE id = 1"
                ).fetchone()
                if metadata is None:
                    raise RuntimeError("Missing index metadata")
                old_hash, old_model, old_dim, old_size, old_count = metadata
                if (
                    old_hash != corpus_hash
                    or old_model != embedding_model
                    or old_dim != dimension
                    or old_size != CHUNK_SIZE
                ):
                    raise RuntimeError(
                        "Existing index uses a different corpus, "
                        "model, dimension, or chunk protocol. "
                        "Pass --rebuild explicitly."
                    )
                return {
                    "chunks": int(old_count),
                    "dimension": int(old_dim),
                    "already_indexed": True,
                }
            conn.execute(f"""
                CREATE TABLE lbrag_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    char_start INTEGER NOT NULL,
                    char_end INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    embedding vector({dimension}) NOT NULL,
                    CHECK (char_start >= 0),
                    CHECK (char_end > char_start)
                )
            """)
            conn.execute("""
                CREATE TABLE lbrag_meta (
                    id SMALLINT PRIMARY KEY CHECK (id = 1),
                    corpus_hash TEXT NOT NULL,
                    embedding_model TEXT NOT NULL,
                    dimension INTEGER NOT NULL,
                    chunk_size INTEGER NOT NULL,
                    chunk_count BIGINT NOT NULL
                )
            """)

            def insert_batch(
                batch: list[Any],
                vectors: list[list[float]],
            ) -> None:
                nonlocal count
                rows = [
                    (i[3], i[0], i[1], i[2], i[4], vector_literal(v))
                    for i, v in zip(batch, vectors, strict=True)
                ]
                with conn.cursor() as cursor:
                    cursor.executemany(
                        "INSERT INTO lbrag_chunks (chunk_id, file_path, "
                        "char_start, char_end, content, embedding) "
                        "VALUES (%s, %s, %s, %s, %s, %s::vector)",
                        rows,
                    )
                count += len(rows)

            insert_batch(first_batch, first_vectors)
            for batch in batches(stream, batch_size):
                vectors, current_dimension = embed_batch(batch)
                if current_dimension != dimension:
                    raise ValueError("Embedding dimension changed during indexing")
                insert_batch(batch, vectors)
            conn.execute(
                "INSERT INTO lbrag_meta "
                "(id, corpus_hash, embedding_model, dimension, "
                "chunk_size, chunk_count) "
                "VALUES (1, %s, %s, %s, %s, %s)",
                (
                    corpus_hash,
                    embedding_model,
                    dimension,
                    CHUNK_SIZE,
                    count,
                ),
            )
            if dimension <= 2000:
                conn.execute(
                    "CREATE INDEX lbrag_embedding_hnsw "
                    "ON lbrag_chunks "
                    "USING hnsw (embedding vector_cosine_ops)"
                )
            else:
                conn.execute(
                    f"CREATE INDEX lbrag_embedding_hnsw "
                    f"ON lbrag_chunks "
                    f"USING hnsw ((embedding::halfvec({dimension})) "
                    f"halfvec_cosine_ops)"
                )
    return {
        "chunks": count,
        "dimension": dimension,
        "already_indexed": False,
    }


def load_precomputed_index(
    *,
    artifacts_root: Path,
    dsn: str,
    rebuild: bool = False,
) -> dict[str, Any]:
    """Load chunk+embedding parquet into lbrag_chunks. Zero embedding."""
    import psycopg
    import pyarrow.parquet as pq

    parquet = artifacts_root / "chunks.parquet"
    manifest = json.loads((artifacts_root / "manifest.json").read_text())

    table = pq.read_table(parquet)
    if table.num_rows != manifest["chunks"]:
        raise RuntimeError(f"artifact row mismatch: {table.num_rows} vs {manifest['chunks']}")

    rows = [
        (
            r["chunk_id"],
            r["file_path"],
            r["char_start"],
            r["char_end"],
            r["content"],
            "[" + ",".join(f"{x:.6g}" for x in r["embedding"]) + "]",
        )
        for r in table.to_pylist()
    ]

    with psycopg.connect(dsn) as conn, conn.transaction():
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        if rebuild:
            conn.execute("DROP TABLE IF EXISTS lbrag_chunks CASCADE")
            conn.execute("DROP TABLE IF EXISTS lbrag_meta CASCADE")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS lbrag_chunks (
                chunk_id TEXT PRIMARY KEY,
                file_path TEXT NOT NULL,
                char_start INTEGER NOT NULL,
                char_end INTEGER NOT NULL,
                content TEXT NOT NULL,
                embedding vector(1024) NOT NULL
            )
        """)
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO lbrag_chunks VALUES "
                "(%s, %s, %s, %s, %s, %s::vector) "
                "ON CONFLICT (chunk_id) DO NOTHING",
                rows,
            )
        conn.execute("""
            CREATE TABLE IF NOT EXISTS lbrag_meta (
                id SMALLINT PRIMARY KEY CHECK (id = 1),
                corpus_hash TEXT,
                embedding_model TEXT,
                dimension INTEGER,
                chunk_size INTEGER,
                chunk_count BIGINT
            )
        """)
        conn.execute(
            "INSERT INTO lbrag_meta VALUES "
            "(1, %s, %s, 1024, %s, %s) "
            "ON CONFLICT (id) DO UPDATE SET "
            "corpus_hash = EXCLUDED.corpus_hash, "
            "chunk_count = EXCLUDED.chunk_count",
            (
                manifest["corpus_hash"],
                manifest["embedding_model"],
                manifest["chunk_size"],
                manifest["chunks"],
            ),
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS lbrag_embedding_hnsw "
            "ON lbrag_chunks "
            "USING hnsw (embedding vector_cosine_ops)"
        )
    return {"chunks": len(rows), "from_artifacts": True}


class PrecomputedQueryEmbedder:
    """Returns vectors from a precomputed parquet; never loads a model."""

    def __init__(self, artifacts_root: Path) -> None:
        import numpy as np
        import pyarrow.parquet as pq

        table = pq.read_table(artifacts_root / "queries.parquet")
        self._lookup: dict[str, Any] = {
            r["query"]: np.asarray(r["embedding"], dtype=np.float32) for r in table.to_pylist()
        }

    def embed_query(self, text: str) -> list[float]:
        vec = self._lookup.get(text)
        if vec is None:
            raise KeyError(f"query not in precomputed set: {text[:60]!r}")
        return [float(x) for x in vec]


class LegalBenchRuntime:
    def __init__(
        self,
        *,
        dsn: str,
        corpus_root: Path,
        embedder: Any,
        embedding_model: str,
        exact_baseline: bool = False,
    ) -> None:
        import psycopg

        self.conn = psycopg.connect(dsn, autocommit=True)
        self.corpus_root = corpus_root
        self.embedder = embedder
        self.embedding_model = embedding_model
        self.exact_baseline = exact_baseline

        metadata = self.conn.execute(
            "SELECT corpus_hash, embedding_model, dimension, chunk_size "
            "FROM lbrag_meta WHERE id = 1"
        ).fetchone()
        if not metadata:
            raise RuntimeError("LegalBench index is not prepared")
        (
            self.expected_hash,
            stored_model,
            self.dimension,
            chunk_size,
        ) = metadata
        if stored_model != embedding_model:
            raise RuntimeError("Embedding model does not match the stored index")
        if chunk_size != CHUNK_SIZE:
            raise RuntimeError("Chunking protocol mismatch")
        if corpus_fingerprint(corpus_root) != self.expected_hash:
            raise RuntimeError("LegalBench corpus changed after indexing")

    def corpus_fingerprint(self) -> str:
        return self.expected_hash

    def retrieve(self, query: str, *, top_k: int) -> dict[str, Any]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        import time

        vector = validate_vector(
            self.embedder.embed_query(query, model=self.embedding_model),
            expected_dimension=int(self.dimension),
        )
        dimension = int(self.dimension)
        if dimension <= 2000:
            expression = "embedding"
            input_cast = "vector"
        else:
            expression = f"embedding::halfvec({dimension})"
            input_cast = "halfvec"
        sql = f"""
            SELECT chunk_id, file_path, char_start, char_end
            FROM lbrag_chunks
            ORDER BY {expression} <=> %s::{input_cast}
            LIMIT %s
        """
        started = time.perf_counter()
        with self.conn.transaction():
            exact = "off" if self.exact_baseline else "on"
            self.conn.execute("SELECT set_config('enable_indexscan', %s, true)", (exact,))
            self.conn.execute(
                "SELECT set_config('enable_indexonlyscan', %s, true)",
                (exact,),
            )
            self.conn.execute(
                "SELECT set_config('enable_bitmapscan', %s, true)",
                (exact,),
            )
            self.conn.execute(
                "SELECT set_config('hnsw.iterative_scan', %s, true)",
                ("off" if self.exact_baseline else "relaxed_order",),
            )
            self.conn.execute(
                "SELECT set_config('hnsw.ef_search', %s, true)",
                (str(min(1000, max(40, top_k * 4))),),
            )
            rows = self.conn.execute(sql, (vector_literal(vector), top_k)).fetchall()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return {
            "hits": [
                {
                    "chunk_id": row[0],
                    "doc": row[1],
                    "start": row[2],
                    "end": row[3],
                }
                for row in rows
            ],
            "arms_used": ["dense_exact" if self.exact_baseline else "dense_hnsw"],
            "arm_diagnostics": [
                {
                    "name": "dense",
                    "hits": len(rows),
                    "latency_ms": elapsed_ms,
                    "error": None,
                }
            ],
        }

    def close(self) -> None:
        self.conn.close()


def build_runtime(
    *,
    suite: str,
    config_name: str,
    exact_baseline: bool,
    no_graph: bool,
    rrf_k: int | None,
    overrides: dict,
) -> LegalBenchRuntime:
    del config_name, no_graph, rrf_k, overrides
    if suite != "legalbench":
        raise ValueError("LegalBenchRuntime requires suite='legalbench'")

    dsn = os.getenv("LEGALBENCH_DSN")
    embedding_model = os.getenv("LEGALBENCH_EMBEDDING_MODEL")
    if not dsn or not embedding_model:
        raise RuntimeError("Set LEGALBENCH_DSN and LEGALBENCH_EMBEDDING_MODEL")

    corpus_root = Path(os.getenv("LEGALBENCH_CORPUS", "data/legalbench/corpus")).resolve()

    artifacts = os.getenv("LEGALBENCH_ARTIFACTS")
    embedder: Any
    if artifacts and (Path(artifacts) / "queries.parquet").is_file():
        embedder = PrecomputedQueryEmbedder(Path(artifacts))
    else:
        embedder_spec = os.getenv("LEGALBENCH_EMBEDDER")
        if not embedder_spec:
            raise RuntimeError("Set LEGALBENCH_EMBEDDER or LEGALBENCH_ARTIFACTS")
        embedder = load_embedder(embedder_spec)

    return LegalBenchRuntime(
        dsn=dsn,
        corpus_root=corpus_root,
        embedder=embedder,
        embedding_model=embedding_model,
        exact_baseline=exact_baseline,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare LegalBench-RAG index and gold.")
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("data/legalbench/corpus"),
    )
    parser.add_argument(
        "--benchmarks",
        type=Path,
        default=Path("data/legalbench/benchmarks"),
    )
    parser.add_argument("--dsn", default=os.getenv("LEGALBENCH_DSN"))
    parser.add_argument(
        "--embedder",
        default=os.getenv("LEGALBENCH_EMBEDDER"),
    )
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("LEGALBENCH_EMBEDDING_MODEL"),
    )
    parser.add_argument(
        "--dataset-revision",
        required=True,
        help="Immutable dataset revision, tag, or archive checksum.",
    )
    parser.add_argument("--subset", choices=SUBSETS, action="append")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--gold-out",
        type=Path,
        default=Path("src/evals/gold/legalbench.jsonl"),
    )
    parser.add_argument("--manifest-out", type=Path)
    parser.add_argument("--rebuild", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if not args.dsn:
            raise ValueError("Missing LegalBench DSN")
        if not args.embedder or not args.embedding_model:
            raise ValueError("Missing embedding configuration")

        corpus = args.corpus.resolve()
        benchmarks = args.benchmarks.resolve()
        subsets = args.subset or list(SUBSETS)

        fingerprint = corpus_fingerprint(corpus)
        embedder = load_embedder(args.embedder)
        stats = build_postgres_index(
            corpus_root=corpus,
            dsn=args.dsn,
            embedder=embedder,
            embedding_model=args.embedding_model,
            corpus_hash=fingerprint,
            batch_size=args.batch_size,
            rebuild=args.rebuild,
        )
        gold = load_benchmark_gold(
            corpus_root=corpus,
            benchmarks_root=benchmarks,
            subsets=subsets,
            corpus_hash=fingerprint,
            dataset_revision=args.dataset_revision,
        )
        write_jsonl(args.gold_out, gold)

        manifest = {
            "schema_version": 1,
            "suite": "legalbench",
            "dataset_revision": args.dataset_revision,
            "protocol": PROTOCOL_VERSION,
            "chunk_size": CHUNK_SIZE,
            "overlap": 0,
            "corpus_hash": fingerprint,
            "embedding_model": args.embedding_model,
            "subsets": subsets,
            "question_count": len(gold),
            "index": stats,
        }
        manifest_path = args.manifest_out or args.gold_out.with_name(
            args.gold_out.stem + ".manifest.json"
        )
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(manifest, indent=2))
        print(f"Gold: {args.gold_out}")
        print(f"Manifest: {manifest_path}")
        return 0
    except Exception as exc:
        print(
            f"LEGALBENCH PREPARATION FAILED: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
