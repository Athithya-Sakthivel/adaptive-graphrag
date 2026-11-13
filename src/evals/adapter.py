"""Adapter from the eval framework to the production retriever.

Implements the eval-framework runtime contract:
    build_runtime(*, suite, config_name, exact_baseline, no_graph,
                  rrf_k, overrides)
    -> EvalRuntime(corpus_fingerprint(), retrieve(query, top_k), close())

The production retrieval path (retriever.route) is what gets
measured. QueryEmbedder is loaded once per process; multiple ablation
conditions share a single instance to avoid repeated cold starts.

Arm filtering uses monkeypatch — route() has no arm-whitelist
parameter, and ablations need to disable specific arms. This is the
only pragmatic way to test "dense alone" or "graph alone" without
forking the router.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import psycopg

from retriever import Filters, dense, exact, graph, route, sparse
from retriever.config import RetrieverConfig, default_config

from .fingerprint import case_law_fingerprint

_DEFAULT_MODEL_DIR = Path(
    os.environ.get(
        "RETRIEVER_MODEL_DIR",
        "/workspace/src/indexing_pipeline/embedders",
    )
)

_shared_embedder: Any = None


def _get_shared_embedder() -> Any:
    global _shared_embedder
    if _shared_embedder is None:
        from retriever.embedder import QueryEmbedder

        _shared_embedder = QueryEmbedder(_DEFAULT_MODEL_DIR)
    return _shared_embedder


_ALL_ARMS = frozenset({"dense", "sparse", "exact", "graph_1hop", "dense_citers"})


@contextmanager
def _arm_filter(disabled: frozenset[str]) -> Iterator[None]:
    """Temporarily replace disabled arms with no-op functions.

    dense_citers shares dense.search, so it cannot be disabled
    independently; requesting it alone raises rather than silently
    producing un-ablated results.

    Restores originals on exit even if route() raises. Safe for eval
    because eval is single-threaded.
    """
    if "dense_citers" in disabled and "dense" not in disabled:
        raise ValueError(
            "dense_citers shares dense.search; disable 'dense' to ablate both, "
            "or add a separate dense_citers hook before ablating it alone"
        )

    saved: dict[str, Any] = {}
    noop = lambda *a, **k: []
    try:
        if "dense" in disabled:
            saved["dense"] = dense.search
            dense.search = noop
        if "sparse" in disabled:
            saved["sparse"] = sparse.search
            sparse.search = noop
        if "exact" in disabled:
            saved["exact"] = exact.search
            exact.search = noop
        if "graph_1hop" in disabled:
            saved["graph"] = graph.search
            graph.search = noop
        yield
    finally:
        for key, original in saved.items():
            if key == "dense":
                dense.search = original
            elif key == "sparse":
                sparse.search = original
            elif key == "exact":
                exact.search = original
            elif key == "graph":
                graph.search = original


class EvalRuntime:
    """One eval session. Holds connection, shared embedder, config."""

    def __init__(
        self,
        conn: psycopg.Connection,
        config: RetrieverConfig,
        no_graph: bool,
        enabled_arms: frozenset[str] | None,
        corpus_hash: str,
        published_only: bool = False,
    ) -> None:
        self.conn = conn
        self.config = config
        self.no_graph = no_graph
        self.enabled_arms = enabled_arms
        self.published_only = published_only
        self._corpus_hash = corpus_hash
        self.embedder = _get_shared_embedder()

    def corpus_fingerprint(self) -> str:
        return self._corpus_hash

    def _disabled_arms(self) -> frozenset[str]:
        if self.enabled_arms is not None:
            return _ALL_ARMS - self.enabled_arms
        if self.no_graph:
            return frozenset({"graph_1hop"})
        return frozenset()

    def retrieve(self, query: str, *, top_k: int) -> dict:
        if top_k < 1:
            raise ValueError("top_k must be positive")

        disabled = self._disabled_arms()
        with _arm_filter(disabled):
            result = route(
                self.conn,
                self.embedder,
                query,
                filters=Filters(published_only=self.published_only),
                top_k=top_k,
                config=self.config,
            )

        return {
            "hits": [
                {
                    "case_id": h.case_id,
                    "chunk_id": h.chunk_id,
                    "is_parent": h.is_parent,
                    "score": h.score,
                    "rank": h.rank,
                }
                for h in result.hits
            ],
            "arms_used": list(result.arms_used),
            "arm_diagnostics": [
                {
                    "name": d.name,
                    "attempted": d.attempted,
                    "hits": d.hits,
                    "latency_ms": d.latency_ms,
                    "error": d.error,
                }
                for d in result.arm_diagnostics
            ],
            "query_class_actual": result.query_class.value,
        }

    def close(self) -> None:
        self.conn.close()


def _build_config(
    *,
    exact_baseline: bool,
    rrf_k: int | None,
    overrides: dict,
) -> tuple[RetrieverConfig, bool, frozenset[str] | None]:
    """Translate eval-framework params into a RetrieverConfig."""
    config = default_config()
    no_graph = False
    enabled_arms: frozenset[str] | None = None

    if rrf_k is not None:
        config = config.with_overrides(
            dense_k=rrf_k,
            sparse_k_holding=rrf_k,
            sparse_k_lookup=rrf_k,
            sparse_k_validity=rrf_k,
            exact_k=rrf_k,
            graph_k=rrf_k,
            dense_citers_k=rrf_k,
        )

    graph_hops = overrides.get("graph_hops")
    if graph_hops == 0:
        no_graph = True

    arms = overrides.get("enabled_arms")
    if isinstance(arms, list):
        enabled_arms = frozenset(arms)

    if exact_baseline:
        config = config.with_overrides(
            iterative_scan_mode="off",
            ef_search=1000,
        )

    _metadata_keys = {
        "condition_name",
        "candidate_cap",
        "rrf_enabled",
        "enabled_arms",
        "graph_hops",
    }
    config_overrides = {k: v for k, v in overrides.items() if k not in _metadata_keys}
    if config_overrides:
        config = config.with_overrides(**config_overrides)

    return config, no_graph, enabled_arms


def build_runtime(
    *,
    suite: str,
    config_name: str,
    exact_baseline: bool,
    published_only: bool = False,
    no_graph: bool,
    rrf_k: int | None,
    overrides: dict,
) -> EvalRuntime:
    """Factory called by run_eval.py."""
    if suite != "custom":
        raise ValueError(f"EvalRuntime requires suite='custom', got {suite!r}")

    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        raise RuntimeError("DATABASE_URL must be set")

    conn = psycopg.connect(dsn, autocommit=True)

    config, cfg_no_graph, enabled_arms = _build_config(
        exact_baseline=exact_baseline,
        rrf_k=rrf_k,
        overrides=overrides,
    )

    no_graph = no_graph or cfg_no_graph
    corpus_hash = case_law_fingerprint(conn)

    return EvalRuntime(
        conn=conn,
        config=config,
        no_graph=no_graph,
        enabled_arms=enabled_arms,
        published_only=published_only,
        corpus_hash=corpus_hash,
    )


__all__ = ["EvalRuntime", "build_runtime"]
