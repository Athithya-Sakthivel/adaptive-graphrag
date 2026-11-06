"""Adapter from the eval framework to the production retriever.

Implements the eval-framework contract:

    build_runtime(*, suite, config_name, exact_baseline, no_graph, rrf_k, overrides)
    -> EvalRuntime(corpus_fingerprint(), retrieve(query, top_k), close())

Key design points:

    * Route() executes — no parallel router. The production path is what
      gets measured.
    * QueryEmbedder is loaded once per process (module-level cache);
      ablation's 5 conditions share a single instance, avoiding 5× cold start.
    * Arm filtering uses monkeypatch, not config — route() has no arm-whitelist
      parameter, and the ablation needs to disable specific arms for A_dense.
      This is the only pragmatic way to test "dense alone" without forking the
      router.
    * Exact baseline uses iterative_scan_mode='off' + ef_search=1000, which
      makes the dense arm do an exact scan. The HNSW index is bypassed.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import psycopg

from evals.fingerprint import case_law_fingerprint

from . import dense, exact, graph, sparse
from .config import RetrieverConfig, default_config
from .filters import Filters
from .router import route

_DEFAULT_MODEL_DIR = Path(
    "/workspace/src/indexing_pipeline/embedders/dinghy-law-0.6b-v1-openvino-fp16"
)

_shared_embedder: Any = None


def _get_shared_embedder() -> Any:
    """Load QueryEmbedder once; share across ablation conditions."""
    global _shared_embedder
    if _shared_embedder is None:
        from .embedder import QueryEmbedder

        _shared_embedder = QueryEmbedder(_DEFAULT_MODEL_DIR)
    return _shared_embedder


_ALL_ARMS = frozenset({"dense", "sparse", "exact", "graph_1hop", "graph_2hop"})


@contextmanager
def _arm_filter(disabled: frozenset[str]) -> Iterator[None]:
    """Temporarily replace disabled arms with no-op functions.

    Restores originals on exit, even if route() raises. Safe for eval
    because eval is single-threaded (no other request can see the
    patched module while this block is active).
    """
    saved: dict[str, Any] = {}
    try:
        if "dense" in disabled:
            saved["dense"] = dense.search
            dense.search = lambda *a, **k: []  # type: ignore[assignment]
        if "sparse" in disabled:
            saved["sparse"] = sparse.search
            sparse.search = lambda *a, **k: []  # type: ignore[assignment]
        if "exact" in disabled:
            saved["exact"] = exact.search
            exact.search = lambda *a, **k: []  # type: ignore[assignment]
        if {"graph_1hop", "graph_2hop"} & disabled:
            saved["graph"] = graph.search
            graph.search = lambda *a, **k: []  # type: ignore[assignment]
        yield
    finally:
        if "dense" in saved:
            dense.search = saved["dense"]
        if "sparse" in saved:
            sparse.search = saved["sparse"]
        if "exact" in saved:
            exact.search = saved["exact"]
        if "graph" in saved:
            graph.search = saved["graph"]


class EvalRuntime:
    """One eval session. Holds connection, embedder (shared), config."""

    def __init__(
        self,
        conn: psycopg.Connection,
        config: RetrieverConfig,
        no_graph: bool,
        enabled_arms: frozenset[str] | None,
        corpus_hash: str,
    ) -> None:
        self.conn = conn
        self.config = config
        self.no_graph = no_graph
        self.enabled_arms = enabled_arms
        self._corpus_hash = corpus_hash
        self.embedder = _get_shared_embedder()

    def corpus_fingerprint(self) -> str:
        return self._corpus_hash

    def _disabled_arms(self) -> frozenset[str]:
        """Compute the set of arms to disable for this runtime."""
        if self.enabled_arms is not None:
            # Strict whitelist: disable everything not listed
            return _ALL_ARMS - self.enabled_arms
        if self.no_graph:
            return frozenset({"graph_1hop", "graph_2hop"})
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
                filters=Filters(),
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
        # Embedder is shared; do NOT close it here.
        self.conn.close()


def _build_config(
    *,
    exact_baseline: bool,
    rrf_k: int | None,
    overrides: dict,
) -> tuple[RetrieverConfig, bool, frozenset[str] | None]:
    """Translate eval-framework params into a RetrieverConfig.

    Returns (config, no_graph, enabled_arms).
    """
    config = default_config()
    no_graph = False
    enabled_arms: frozenset[str] | None = None

    # --- RRF k override: applies to every per-arm k ---
    if rrf_k is not None:
        config = config.with_overrides(
            dense_k=rrf_k,
            sparse_k_holding=rrf_k,
            sparse_k_lookup=rrf_k,
            sparse_k_validity=rrf_k,
            sparse_k_application=rrf_k,
            exact_k=rrf_k,
            graph_k=rrf_k,
            dense_citers_k=rrf_k,
            seeds_pass2_k=rrf_k,
            graph_pass2_k=rrf_k,
        )

    # --- Graph depth from ablation overrides ---
    graph_hops = overrides.get("graph_hops")
    if graph_hops == 0:
        no_graph = True
        config = config.with_overrides(application_2hop_enabled=False)
    elif graph_hops == 1:
        config = config.with_overrides(application_2hop_enabled=False)
    elif graph_hops == 2:
        config = config.with_overrides(application_2hop_enabled=True)

    # --- Arm whitelist (A_dense uses this) ---
    arms = overrides.get("enabled_arms")
    if isinstance(arms, list):
        enabled_arms = frozenset(arms)

    # --- Exact baseline: disable HNSW, force exact scan ---
    if exact_baseline:
        config = config.with_overrides(
            iterative_scan_mode="off",
            ef_search=1000,
        )

    # --- Remaining RetrieverConfig fields via overrides ---
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
    no_graph: bool,
    rrf_k: int | None,
    overrides: dict,
) -> EvalRuntime:
    """Factory called by run_eval.py and ablation.py."""
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

    # The factory's no_graph param and the config's no_graph must agree.
    no_graph = no_graph or cfg_no_graph

    corpus_hash = case_law_fingerprint(conn)

    # Silence the unused config_name warning
    del config_name

    return EvalRuntime(
        conn=conn,
        config=config,
        no_graph=no_graph,
        enabled_arms=enabled_arms,
        corpus_hash=corpus_hash,
    )
