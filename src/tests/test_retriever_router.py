"""Unit tests for adaptive retrieval routing.

These tests mock dense, sparse, exact, and graph retrieval. They test
routing decisions and orchestration, not SQL execution.

Parent expansion lives in exact.case_ids_to_hits (not graph), so
install_parent_expansion patches router.exact.

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

EMBEDDING_DIMENSIONS = 384


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class FakeTokenizer:
    def encode(self, text: str, **kwargs: Any) -> list[int]:
        return list(range(max(1, len(text.split()))))


class FakeEmbedder:
    """Deterministic embedder that does not load a real model."""

    def __init__(self) -> None:
        self.tokenizer = FakeTokenizer()
        self.max_seq_length = 128
        self.calls: list[str] = []

    def embed_query(self, query: str) -> np.ndarray:
        self.calls.append(query)
        return np.ones(EMBEDDING_DIMENSIONS, dtype=np.float32)


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
    """Return one deterministic parent Hit per requested case ID.

    Patches exact.case_ids_to_hits — the router's _expand_parents helper
    calls it there, not on the graph module.
    """

    def fake_case_ids_to_hits(
        conn: Any,
        case_ids: list[int],
        *args: Any,
        **kwargs: Any,
    ) -> list[Hit]:
        return [make_hit(f"parent-{case_id}", case_id, is_parent=True) for case_id in case_ids]

    monkeypatch.setattr(router.exact, "case_ids_to_hits", fake_case_ids_to_hits)


def install_opinion_id_resolution(
    monkeypatch: pytest.MonkeyPatch,
    case_ids: list[int],
) -> None:
    monkeypatch.setattr(
        router.exact,
        "resolve_opinion_id",
        lambda *args, **kwargs: list(case_ids),
    )


# ---------------------------------------------------------------------------
# HOLDING
# ---------------------------------------------------------------------------


def test_holding_uses_dense_and_sparse_without_graph(
    monkeypatch: pytest.MonkeyPatch,
    embedder: FakeEmbedder,
) -> None:
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

    result = router.route(None, embedder, "what did the court hold on qualified immunity")

    assert result.query_class is QueryClass.HOLDING
    assert "dense" in calls
    assert "sparse" in calls
    assert "graph_1hop" not in result.arms_used
    assert embedder.calls == ["what did the court hold on qualified immunity"]
    assert result.hits


def test_holding_survives_dense_failure(
    monkeypatch: pytest.MonkeyPatch,
    embedder: FakeEmbedder,
) -> None:
    """A failing dense arm must degrade to sparse, not abort the query."""

    def broken_dense(*args: Any, **kwargs: Any) -> list[Hit]:
        raise RuntimeError("index unavailable")

    monkeypatch.setattr(router.dense, "search", broken_dense)
    monkeypatch.setattr(
        router.sparse,
        "search",
        lambda *args, **kwargs: [make_hit("sparse-1", case_id=2)],
    )

    result = router.route(None, embedder, "what did the court hold on qualified immunity")

    assert result.hits
    dense_diag = next(d for d in result.arm_diagnostics if d.name == "dense")
    assert dense_diag.error is not None
    assert dense_diag.hits == 0


def test_citation_precedes_holding_keyword(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rule 2 (citation) fires before rule 4 (holding keyword).

    'what did 410 U.S. 113 hold' resolves to LOOKUP, not HOLDING.
    Documented limitation: citation presence wins over holding intent.
    """
    install_parent_expansion(monkeypatch)
    monkeypatch.setattr(router.sparse, "search", lambda *a, **k: [make_hit("s", 2)])
    monkeypatch.setattr(router.exact, "search", lambda *a, **k: [make_hit("e", 3, is_parent=True)])
    monkeypatch.setattr(router.graph, "search", lambda *a, **k: [])

    result = router.route(None, None, "what did 410 U.S. 113 hold")

    assert result.query_class is QueryClass.LOOKUP
    assert "exact" in result.arms_used


def test_holding_skips_exact_arm_without_citation(
    monkeypatch: pytest.MonkeyPatch,
    embedder: FakeEmbedder,
) -> None:
    """HOLDING with a case name but no reporter citation runs dense+sparse only."""
    monkeypatch.setattr(router.dense, "search", lambda *a, **k: [make_hit("d", 1)])
    monkeypatch.setattr(router.sparse, "search", lambda *a, **k: [make_hit("s", 2)])

    def forbidden_exact(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("HOLDING without a citation must not run the exact arm")

    monkeypatch.setattr(router.exact, "search", forbidden_exact)

    result = router.route(None, embedder, "what did the court hold in Roe v. Wade")

    assert result.query_class is QueryClass.HOLDING
    assert "exact" not in result.arms_used


# ---------------------------------------------------------------------------
# LOOKUP
# ---------------------------------------------------------------------------


def test_lookup_resolves_target_and_uses_incoming_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
            ("graph", {"seed_case_ids": seed_case_ids, "direction": direction, "depth": depth})
        )
        return [{"case_id": 20, "hop": 1}]

    monkeypatch.setattr(router.sparse, "search", fake_sparse)
    monkeypatch.setattr(router.graph, "search", fake_graph)

    result = router.route(None, None, "which cases cite Roe v. Wade")

    assert result.query_class is QueryClass.LOOKUP

    graph_calls = [payload for name, payload in calls if name == "graph"]
    assert len(graph_calls) == 1
    assert graph_calls[0]["seed_case_ids"] == [10]
    assert graph_calls[0]["direction"] == "incoming"
    assert graph_calls[0]["depth"] == 1

    assert any(name == "sparse" for name, _ in calls)
    assert "graph_1hop" in result.arms_used
    assert result.hits


def test_lookup_never_loads_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    """LOOKUP is a structural query; no embedding model may be touched."""
    embedder = FakeEmbedder()

    install_case_resolution(monkeypatch, [10])
    install_parent_expansion(monkeypatch)
    monkeypatch.setattr(router.sparse, "search", lambda *a, **k: [])
    monkeypatch.setattr(router.graph, "search", lambda *a, **k: [{"case_id": 20, "hop": 1}])

    router.route(None, embedder, "which cases cite Roe v. Wade")

    assert embedder.calls == []


def test_lookup_capitalized_template_resolves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The gold-set template 'Which cases cite X?' must resolve X."""
    install_case_resolution(monkeypatch, [10])
    install_parent_expansion(monkeypatch)
    monkeypatch.setattr(router.sparse, "search", lambda *a, **k: [])
    monkeypatch.setattr(router.graph, "search", lambda *a, **k: [{"case_id": 20, "hop": 1}])

    result = router.route(None, None, "Which cases cite Delevie v. State?")

    assert "graph_1hop" in result.arms_used
    resolution = next(d for d in result.arm_diagnostics if d.name == "case_name_resolution")
    assert resolution.hits == 1


def test_lookup_uses_explicit_opinion_id(monkeypatch: pytest.MonkeyPatch) -> None:
    install_opinion_id_resolution(monkeypatch, [10])
    install_parent_expansion(monkeypatch)
    monkeypatch.setattr(router.sparse, "search", lambda *a, **k: [])
    monkeypatch.setattr(router.graph, "search", lambda *a, **k: [{"case_id": 20, "hop": 1}])

    def forbidden_name_resolution(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Explicit opinion ID must bypass case-name resolution")

    monkeypatch.setattr(router.exact, "resolve_case_ids", forbidden_name_resolution)

    result = router.route(None, None, "Find cases citing opinion 123456")

    assert result.query_class is QueryClass.LOOKUP
    assert "graph_1hop" in result.arms_used
    assert any(d.name == "opinion_id_resolution" for d in result.arm_diagnostics)


def test_lookup_skips_graph_when_target_cannot_be_resolved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_case_resolution(monkeypatch, [])
    monkeypatch.setattr(
        router.sparse,
        "search",
        lambda *args, **kwargs: [make_hit("sparse-unrelated", case_id=99)],
    )

    def forbidden_graph(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Graph traversal must be skipped when the target is unresolved")

    monkeypatch.setattr(router.graph, "search", forbidden_graph)

    result = router.route(None, None, "which cases cite Unknown v. Unknown")

    assert result.query_class is QueryClass.LOOKUP
    assert "graph_1hop" not in result.arms_used
    assert result.hits


def test_lookup_graph_hits_carry_hop_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    install_case_resolution(monkeypatch, [10])
    install_parent_expansion(monkeypatch)
    monkeypatch.setattr(router.sparse, "search", lambda *a, **k: [])
    monkeypatch.setattr(router.graph, "search", lambda *a, **k: [{"case_id": 20, "hop": 1}])

    result = router.route(None, None, "which cases cite Roe v. Wade", top_k=10)

    graph_hit = next(h for h in result.hits if h.case_id == 20)
    assert graph_hit.meta["hop"] == 1
    assert graph_hit.meta["graph_direction"] == "incoming"


# ---------------------------------------------------------------------------
# VALIDITY
# ---------------------------------------------------------------------------


def test_validity_searches_incoming_citers_then_filters_dense_search(
    monkeypatch: pytest.MonkeyPatch,
    embedder: FakeEmbedder,
) -> None:
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
        graph_calls.append({"seed_case_ids": seed_case_ids, "direction": direction, "depth": depth})
        return [{"case_id": 20, "hop": 1}]

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

    result = router.route(None, embedder, "has Roe v. Wade been overruled")

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
    install_case_resolution(monkeypatch, [10])
    install_parent_expansion(monkeypatch)

    monkeypatch.setattr(
        router.sparse,
        "search",
        lambda *args, **kwargs: [make_hit("sparse-target", case_id=10)],
    )
    monkeypatch.setattr(router.graph, "search", lambda *args, **kwargs: [])

    def forbidden_dense(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Dense citers search must be skipped when there are no citers")

    monkeypatch.setattr(router.dense, "search", forbidden_dense)

    result = router.route(None, embedder, "has Roe v. Wade been overruled")

    assert result.query_class is QueryClass.VALIDITY
    assert "dense_citers" not in result.arms_used


def test_validity_falls_back_when_target_unresolved(
    monkeypatch: pytest.MonkeyPatch,
    embedder: FakeEmbedder,
) -> None:
    install_case_resolution(monkeypatch, [])
    monkeypatch.setattr(
        router.sparse,
        "search",
        lambda *args, **kwargs: [make_hit("sparse-1", case_id=1)],
    )

    def forbidden_graph(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Graph must be skipped when the target is unresolved")

    monkeypatch.setattr(router.graph, "search", forbidden_graph)

    result = router.route(None, embedder, "has Unknown v. Unknown been overruled")

    assert result.query_class is QueryClass.VALIDITY
    assert "graph_1hop" not in result.arms_used
    assert result.hits


def test_validity_survives_dense_failure(
    monkeypatch: pytest.MonkeyPatch,
    embedder: FakeEmbedder,
) -> None:
    install_case_resolution(monkeypatch, [10])
    install_parent_expansion(monkeypatch)
    monkeypatch.setattr(router.sparse, "search", lambda *a, **k: [])
    monkeypatch.setattr(router.graph, "search", lambda *a, **k: [{"case_id": 20, "hop": 1}])

    def broken_dense(*args: Any, **kwargs: Any) -> list[Hit]:
        raise RuntimeError("index unavailable")

    monkeypatch.setattr(router.dense, "search", broken_dense)

    result = router.route(None, embedder, "has Roe v. Wade been overruled")

    assert result.hits
    dense_diag = next(d for d in result.arm_diagnostics if d.name == "dense_citers")
    assert dense_diag.error is not None


# ---------------------------------------------------------------------------
# Shared validation, diagnostics, and filter behavior
# ---------------------------------------------------------------------------


def test_every_attempted_arm_has_a_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
    embedder: FakeEmbedder,
) -> None:
    monkeypatch.setattr(router.dense, "search", lambda *a, **k: [make_hit("d", 1)])
    monkeypatch.setattr(router.sparse, "search", lambda *a, **k: [make_hit("s", 2)])

    result = router.route(None, embedder, "what did the court hold on qualified immunity")

    names = {d.name for d in result.arm_diagnostics}
    assert {"dense", "sparse"} <= names
    assert all(d.latency_ms >= 0 for d in result.arm_diagnostics)


def test_case_diversity_collapses_repeated_cases(
    monkeypatch: pytest.MonkeyPatch,
    embedder: FakeEmbedder,
) -> None:
    """HOLDING allows one chunk per case, so repeats must collapse."""
    monkeypatch.setattr(
        router.dense,
        "search",
        lambda *a, **k: [make_hit("c1", 7), make_hit("c2", 7), make_hit("c3", 8)],
    )
    monkeypatch.setattr(router.sparse, "search", lambda *a, **k: [])

    result = router.route(None, embedder, "what did the court hold on qualified immunity")

    case_ids = [h.case_id for h in result.hits]
    assert len(case_ids) == len(set(case_ids))


def test_empty_case_id_filter_returns_no_results_without_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    assert result.arms_used == []
    assert embedder.calls == []


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_blank_query_raises_value_error(query: str) -> None:
    with pytest.raises(ValueError, match="query"):
        router.route(None, FakeEmbedder(), query)


@pytest.mark.parametrize("top_k", [0, -1])
def test_invalid_top_k_raises(top_k: int) -> None:
    with pytest.raises(ValueError, match="top_k"):
        router.route(None, FakeEmbedder(), "what did the court hold", top_k=top_k)


def test_non_string_query_rejected() -> None:
    with pytest.raises(TypeError):
        router.route(None, FakeEmbedder(), None)  # type: ignore[arg-type]


def test_invalid_filters_type_rejected() -> None:
    with pytest.raises(TypeError):
        router.route(None, FakeEmbedder(), "what did the court hold", filters="court=ca9")  # type: ignore[arg-type]


def test_top_k_limits_returned_hits(
    monkeypatch: pytest.MonkeyPatch,
    embedder: FakeEmbedder,
) -> None:
    monkeypatch.setattr(
        router.dense,
        "search",
        lambda *a, **k: [make_hit(f"c{i}", i + 1) for i in range(30)],
    )
    monkeypatch.setattr(router.sparse, "search", lambda *a, **k: [])

    result = router.route(None, embedder, "what did the court hold on qualified immunity", top_k=5)

    assert len(result.hits) == 5
