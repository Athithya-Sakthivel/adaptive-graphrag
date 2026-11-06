"""Bridges retriever.embedder (query-only) and indexing_pipeline.embedder
(batch) into the embed_documents / embed_query contract that the LegalBench
adapter expects.

Both embedders are cached at module level — LegalBench indexing calls
embed_documents thousands of times; loading the batch embedder per call
would make the index build take weeks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

_doc_embedder: Any = None
_query_embedder: Any = None

_DEFAULT_DEVICE = "CPU"
_DEFAULT_THREADS = 4
_DEFAULT_BATCH_SIZE = 16
_DEFAULT_SEQ = 464


def _get_doc_embedder(model: str) -> Any:
    global _doc_embedder
    if _doc_embedder is None:
        from indexing_pipeline.embedder import Embedder

        _doc_embedder = Embedder(
            model_dir=Path(model),
            max_seq_length=_DEFAULT_SEQ,
            threads=_DEFAULT_THREADS,
            batch_size=_DEFAULT_BATCH_SIZE,
            device=_DEFAULT_DEVICE,
        )
    return _doc_embedder


def _get_query_embedder(model: str) -> Any:
    global _query_embedder
    if _query_embedder is None:
        from retriever.embedder import QueryEmbedder

        _query_embedder = QueryEmbedder(Path(model))
    return _query_embedder


def embed_documents(texts: list[str], *, model: str) -> list[list[float]]:
    """Batch-embed documents for LegalBench indexing."""
    embedder = _get_doc_embedder(model)
    result = embedder.embed_documents(texts)
    return [list(map(float, vec)) for vec in result]


def embed_query(query: str, *, model: str) -> list[float]:
    """Single-query embedding for LegalBench runtime."""
    embedder = _get_query_embedder(model)
    vec = embedder.embed_query(query)
    return list(map(float, vec))
