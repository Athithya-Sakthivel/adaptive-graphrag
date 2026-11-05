"""Unit tests for adaptive retrieval routing.

These tests mock dense, sparse, exact, and graph retrieval. They test
routing decisions and orchestration, not SQL execution.

Run from /workspace/src:

    pytest tests/test_retriever_router.py -v
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from retriever import router
from retriever.classify import QueryClass
from retriever.filters import Filters
from retriever.fusion import Hit

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class FakeTokenizer:
    """Small tokenizer substitute for router-side query-length validation."""

    def encode(self, text: str, **kwargs: Any) -> list[int]:
        return list(range(max(1, len(text.split()))))


class FakeEmbedder:
    """Deterministic embedder that does not load a real model."""

    def __init__(self) -> None:
        self.tokenizer = FakeTokenizer()
        # Match QueryEmbedder's default 128-token limit so length
        # guards are tested realistically.
        self.max_seq_length = 128
        self.calls: list[str] = []

    def embed_query(self, query: str) -> np.ndarray:
        self.calls.append(query)
        return np.ones(1024, dtype=np.float32)


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_hit(
    chunk_id: str,
    case_id: int = 1,
    *,
    is_parent: bool = False,
    score: float = 0.0,
) -> Hit:
    """Create a valid Hit with the fields required by fusion and the CLI."""

    return Hit(
        chunk_id=chunk_id,
        case_id=case_id,
        is_parent=is_parent,
        score=score,
        rank=0,
        meta={
            "preview": f"Test preview for {chunk_id}. " * 8,
            "court": "Test Court",
            "citation": None,
            "date": None,
        },
    )


def install_case_resolution(
    monkeypatch: pytest.MonkeyPatch,
    case_ids: list[int],
) -> None:
    """Stub exact case-name resolution without touching the database."""

    monkeypatch.setattr(
        router.exact,
        "resolve_case_ids",
        lambda *args, **kwargs: list(case_ids),
    )


def install_parent_expansion(monkeypatch: pytest.MonkeyPatch) -> None:
    """Return one deterministic parent Hit for each requested case ID."""

    def fake_case_ids_to_hits(
        conn: Any,
        case_ids: list[int],
        *args: Any,
        **kwargs: Any,
    ) -> list[Hit]:
        return [
            make_hit(
                chunk_id=f"parent-{case_id}",
                case_id=case_id,
                is_parent=True,
            )
            for case_id in case_ids
        ]

    monkeypatch.setattr(
        router.graph,
        "case_ids_to_hits",
        fake_case_ids_to_hits,
    )


# ---------------------------------------------------------------------------
# HOLDING
# ---------------------------------------------------------------------------


def test_holding_uses_dense_and_sparse_without_graph(
    monkeypatch: pytest.MonkeyPatch,
    embedder: FakeEmbedder,
) -> None:
    """HOLDING uses dense + sparse fusion and does not traverse the graph."""

    calls: list[str] = []

    def fake_dense(*args: Any, **kwargs: Any) -> list[Hit]:
        calls.append("dense")
        return [make_hit("dense-1", case_id=1)]

    def fake_sparse(*args: Any, **kwargs: Any) -> list[Hit]:
        calls.append("sparse")
        return [make_hit("sparse-1", case_id=2)]

    def forbidden_graph(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("HOLDING queries must not invoke graph retrieval")

    monkeypatch.setattr(router.dense, "search", fake_dense)
    monkeypatch.setattr(router.sparse, "search", fake_sparse)
    monkeypatch.setattr(router.graph, "search", forbidden_graph)

    result = router.route(
        None,
        embedder,
        "what did the court hold on qualified immunity",
    )

    assert result.query_class is QueryClass.HOLDING
    assert "dense" in calls
    assert "sparse" in calls
    assert "graph" not in calls
    assert embedder.calls == ["what did the court hold on qualified immunity"]
    assert result.hits


# ---------------------------------------------------------------------------
# LOOKUP
# ---------------------------------------------------------------------------


def test_lookup_resolves_target_and_uses_incoming_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LOOKUP finds cases citing the resolved target using incoming edges."""

    calls: list[tuple[str, Any]] = []

    install_case_resolution(monkeypatch, [10])
    install_parent_expansion(monkeypatch)

    def fake_sparse(*args: Any, **kwargs: Any) -> list[Hit]:
        calls.append(("sparse", None))
        return [make_hit("sparse-10", case_id=10)]

    def fake_graph(
        conn: Any,
        seed_case_ids: list[int],
        *,
        direction: str = "incoming",
        depth: int = 1,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        calls.append(
            (
                "graph",
                {
                    "seed_case_ids": seed_case_ids,
                    "direction": direction,
                    "depth": depth,
                },
            )
        )
        return [{"case_id": 20, "hop": 1, "score": 0.0}]

    monkeypatch.setattr(router.sparse, "search", fake_sparse)
    monkeypatch.setattr(router.graph, "search", fake_graph)

    # LOOKUP must not need an embedding model.
    result = router.route(
        None,
        None,
        "which cases cite Roe v. Wade",
    )

    assert result.query_class is QueryClass.LOOKUP

    graph_calls = [payload for name, payload in calls if name == "graph"]
    assert len(graph_calls) == 1
    assert graph_calls[0]["seed_case_ids"] == [10]
    assert graph_calls[0]["direction"] == "incoming"
    assert graph_calls[0]["depth"] == 1

    assert any(name == "sparse" for name, _ in calls)
    assert "graph_1hop" in result.arms_used
    assert result.hits


def test_lookup_skips_graph_when_target_cannot_be_resolved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unresolved case name must not create arbitrary graph seeds."""

    install_case_resolution(monkeypatch, [])

    monkeypatch.setattr(
        router.sparse,
        "search",
        lambda *args, **kwargs: [make_hit("sparse-unrelated", case_id=99)],
    )

    def forbidden_graph(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Graph traversal must be skipped when the target case is unresolved")

    monkeypatch.setattr(router.graph, "search", forbidden_graph)

    result = router.route(
        None,
        None,
        "which cases cite Unknown v. Unknown",
    )

    assert result.query_class is QueryClass.LOOKUP
    assert "graph_1hop" not in result.arms_used


# ---------------------------------------------------------------------------
# VALIDITY
# ---------------------------------------------------------------------------


def test_validity_searches_incoming_citers_then_filters_dense_search(
    monkeypatch: pytest.MonkeyPatch,
    embedder: FakeEmbedder,
) -> None:
    """VALIDITY searches citers, then constrains dense retrieval to those cases."""

    install_case_resolution(monkeypatch, [10])
    install_parent_expansion(monkeypatch)

    graph_calls: list[dict[str, Any]] = []
    dense_filters: list[Filters | None] = []

    monkeypatch.setattr(
        router.sparse,
        "search",
        lambda *args, **kwargs: [make_hit("sparse-target", case_id=10)],
    )

    def fake_graph(
        conn: Any,
        seed_case_ids: list[int],
        *,
        direction: str = "incoming",
        depth: int = 1,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        graph_calls.append(
            {
                "seed_case_ids": seed_case_ids,
                "direction": direction,
                "depth": depth,
            }
        )
        return [{"case_id": 20, "hop": 1, "score": 0.0}]

    def fake_dense(
        conn: Any,
        query_vec: np.ndarray,
        *,
        filters: Filters | None = None,
        **kwargs: Any,
    ) -> list[Hit]:
        dense_filters.append(filters)
        return [make_hit("dense-citer", case_id=20)]

    monkeypatch.setattr(router.graph, "search", fake_graph)
    monkeypatch.setattr(router.dense, "search", fake_dense)

    result = router.route(
        None,
        embedder,
        "has Roe v. Wade been overruled",
    )

    assert result.query_class is QueryClass.VALIDITY

    assert len(graph_calls) == 1
    assert graph_calls[0]["seed_case_ids"] == [10]
    assert graph_calls[0]["direction"] == "incoming"
    assert graph_calls[0]["depth"] == 1

    assert len(dense_filters) == 1
    assert dense_filters[0] is not None
    assert dense_filters[0].case_ids == [20]

    assert embedder.calls
    assert "graph_1hop" in result.arms_used
    assert "dense_citers" in result.arms_used


def test_validity_does_not_run_dense_when_graph_finds_no_citers(
    monkeypatch: pytest.MonkeyPatch,
    embedder: FakeEmbedder,
) -> None:
    """No citing cases means no pointless dense search over an empty set."""

    install_case_resolution(monkeypatch, [10])

    monkeypatch.setattr(
        router.sparse,
        "search",
        lambda *args, **kwargs: [make_hit("sparse-target", case_id=10)],
    )
    monkeypatch.setattr(
        router.graph,
        "search",
        lambda *args, **kwargs: [],
    )

    def forbidden_dense(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Dense citers search must be skipped when the graph has no citers")

    monkeypatch.setattr(router.dense, "search", forbidden_dense)

    result = router.route(
        None,
        embedder,
        "has Roe v. Wade been overruled",
    )

    assert result.query_class is QueryClass.VALIDITY
    # embed_query may be called for the doc, but dense.search must not be.
    assert all("citers" not in c for c in embedder.calls)


# ---------------------------------------------------------------------------
# APPLICATION (case reference)
# ---------------------------------------------------------------------------


def test_application_uses_incoming_graph_and_two_pass_fusion(
    monkeypatch: pytest.MonkeyPatch,
    embedder: FakeEmbedder,
) -> None:
    """APPLICATION resolves the target case and traverses incoming edges.

    Audit fix: pre-hardening APPLICATION used direction="outgoing" and
    seeded from pass1 dense+sparse. Hardened code resolves the named
    case first and uses direction="incoming" — "how has X been applied"
    asks for cases CITING X, not cases X cites.
    """
    graph_calls: list[dict[str, Any]] = []
    pass2_calls: list[tuple[list[Hit], list[Hit]]] = []

    # "Roe v. Wade" resolves to case_id 10 via stubbed exact.resolve_case_ids
    install_case_resolution(monkeypatch, [10])
    install_parent_expansion(monkeypatch)

    # Dense/sparse return hits for case_id 1 (unrelated to the target).
    # These are KEPT in pass1 (case 1 is not the resolved target).
    monkeypatch.setattr(
        router.dense,
        "search",
        lambda *args, **kwargs: [make_hit("dense-seed", case_id=1)],
    )
    monkeypatch.setattr(
        router.sparse,
        "search",
        lambda *args, **kwargs: [make_hit("sparse-seed", case_id=1)],
    )

    def fake_graph(
        conn: Any,
        seed_case_ids: list[int],
        *,
        direction: str = "incoming",
        depth: int = 1,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        graph_calls.append(
            {
                "seed_case_ids": seed_case_ids,
                "direction": direction,
                "depth": depth,
            }
        )
        return [{"case_id": 20, "hop": 1, "score": 0.0}]

    monkeypatch.setattr(router.graph, "search", fake_graph)

    def fake_pass2(
        seeds: list[Hit],
        graph_hits: list[Hit],
        **kwargs: Any,
    ) -> list[Hit]:
        pass2_calls.append((seeds, graph_hits))
        return [make_hit("final-result", case_id=20, is_parent=True)]

    monkeypatch.setattr(router, "rrf_pass2", fake_pass2)

    result = router.route(
        None,
        embedder,
        "how has Roe v. Wade been applied",
    )

    assert result.query_class is QueryClass.APPLICATION

    # Graph called with the RESOLVED seed, incoming direction.
    assert len(graph_calls) == 1
    assert graph_calls[0]["seed_case_ids"] == [10]
    assert graph_calls[0]["direction"] == "incoming"
    assert graph_calls[0]["depth"] == 1

    # Pass 2 fused seeds with graph hits.
    assert len(pass2_calls) == 1
    seeds_at_pass2, graph_at_pass2 = pass2_calls[0]
    # Target case (10) must not appear in seeds passed to pass2.
    assert all(h.case_id != 10 for h in seeds_at_pass2)
    # Graph hits fed to pass2.
    assert len(graph_at_pass2) == 1
    assert graph_at_pass2[0].case_id == 20

    assert result.hits[0].chunk_id == "final-result"


def test_application_falls_back_to_pass1_when_graph_returns_no_citers(
    monkeypatch: pytest.MonkeyPatch,
    embedder: FakeEmbedder,
) -> None:
    """An empty graph result must not discard the dense/sparse pass1."""

    install_case_resolution(monkeypatch, [10])

    monkeypatch.setattr(
        router.dense,
        "search",
        lambda *args, **kwargs: [make_hit("dense-seed", case_id=1)],
    )
    monkeypatch.setattr(
        router.sparse,
        "search",
        lambda *args, **kwargs: [make_hit("sparse-seed", case_id=1)],
    )
    monkeypatch.setattr(
        router.graph,
        "search",
        lambda *args, **kwargs: [],  # target has 0 citers
    )

    def forbidden_pass2(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Pass 2 must be skipped when graph retrieval returns no hits")

    monkeypatch.setattr(router, "rrf_pass2", forbidden_pass2)

    result = router.route(
        None,
        embedder,
        "how has Roe v. Wade been applied",
    )

    assert result.query_class is QueryClass.APPLICATION
    assert result.hits  # pass1 preserved
    assert "dense" in result.arms_used
    assert "sparse" in result.arms_used
    # Graph was attempted (recorded in arms_used) but contributed 0 hits.
    assert "graph_1hop" in result.arms_used


# ---------------------------------------------------------------------------
# APPLICATION (doctrine-only, no case reference)
# ---------------------------------------------------------------------------


def test_doctrine_application_skips_graph(
    monkeypatch: pytest.MonkeyPatch,
    embedder: FakeEmbedder,
) -> None:
    """Doctrine-only APPLICATION (no case ref) routes to dense+sparse only.

    "how has qualified immunity been applied" resolves to APPLICATION
    (via classify rule 5) but has no case reference to seed graph
    traversal. The router must skip the graph arm entirely — no seed
    means no graph.
    """
    monkeypatch.setattr(
        router.dense,
        "search",
        lambda *args, **kwargs: [make_hit("dense-1", case_id=1)],
    )
    monkeypatch.setattr(
        router.sparse,
        "search",
        lambda *args, **kwargs: [make_hit("sparse-1", case_id=1)],
    )

    def forbidden_graph(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Doctrine APPLICATION (no case ref) must not attempt graph")

    monkeypatch.setattr(router.graph, "search", forbidden_graph)

    def forbidden_pass2(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Doctrine APPLICATION must not use pass2 fusion")

    monkeypatch.setattr(router, "rrf_pass2", forbidden_pass2)

    result = router.route(
        None,
        embedder,
        "how has qualified immunity been applied",
    )

    assert result.query_class is QueryClass.APPLICATION
    assert result.hits
    assert "dense" in result.arms_used
    assert "sparse" in result.arms_used
    assert "graph_1hop" not in result.arms_used
    assert "graph_2hop" not in result.arms_used


# ---------------------------------------------------------------------------
# Shared validation and filter behavior
# ---------------------------------------------------------------------------


def test_empty_case_id_filter_returns_no_results_without_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit empty case-ID filter means no eligible cases."""

    embedder = FakeEmbedder()

    def forbidden_search(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Retrieval arms must not run for an empty case-ID filter")

    monkeypatch.setattr(router.dense, "search", forbidden_search)
    monkeypatch.setattr(router.sparse, "search", forbidden_search)
    monkeypatch.setattr(router.exact, "search", forbidden_search)
    monkeypatch.setattr(router.graph, "search", forbidden_search)

    result = router.route(
        None,
        embedder,
        "what did the court hold on qualified immunity",
        filters=Filters(case_ids=[]),
    )

    assert result.hits == []
    assert embedder.calls == []


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_blank_query_raises_value_error(query: str) -> None:
    """Blank input must fail before classification or retrieval."""

    with pytest.raises(ValueError, match="query"):
        router.route(
            None,
            FakeEmbedder(),
            query,
        )
