"""Unit tests for reciprocal-rank fusion and per-case deduplication."""

from __future__ import annotations

import pytest

from retriever.fusion import Hit, case_dedup, rrf_pass1


def mk(chunk_id: str, case_id: int = 1, is_parent: bool = False) -> Hit:
    return Hit(
        chunk_id=chunk_id,
        case_id=case_id,
        is_parent=is_parent,
        meta={"preview": "Legal opinion text"},
    )


def test_single_arm_rrf_math():
    result = rrf_pass1(
        {"dense": [mk("a"), mk("b"), mk("c")]},
        {"dense": 60},
        limit=10,
    )

    assert len(result) == 3
    assert result[0].score == pytest.approx(1 / 61)
    assert result[1].score == pytest.approx(1 / 62)
    assert result[2].score == pytest.approx(1 / 63)


def test_two_arm_shared_hit_boosted():
    result = rrf_pass1(
        {
            "dense": [mk("a"), mk("b")],
            "sparse": [mk("b"), mk("c")],
        },
        {"dense": 60, "sparse": 60},
        limit=10,
    )

    assert result[0].chunk_id == "b"


def test_tie_broken_by_chunk_id():
    result = rrf_pass1(
        {
            "dense": [mk("a"), mk("b")],
            "sparse": [mk("b"), mk("a")],
        },
        {"dense": 60, "sparse": 60},
        limit=10,
    )

    assert result[0].chunk_id == "a"


def test_lower_k_weights_arm_more_heavily():
    result = rrf_pass1(
        {
            "dense": [mk("dense_top", 1), mk("shared", 2)],
            "sparse": [mk("sparse_top", 3), mk("shared", 2)],
        },
        {"dense": 10, "sparse": 1000},
        limit=10,
    )

    assert result[0].chunk_id == "dense_top"


def test_missing_k_raises():
    with pytest.raises(ValueError):
        rrf_pass1({"dense": [mk("a")]}, {"sparse": 60}, limit=10)


def test_zero_k_raises():
    with pytest.raises(ValueError):
        rrf_pass1({"dense": [mk("a")]}, {"dense": 0}, limit=10)


def test_zero_limit_raises():
    with pytest.raises(ValueError):
        rrf_pass1({"dense": [mk("a")]}, {"dense": 60}, limit=0)


def test_empty_arms():
    assert rrf_pass1({}, {}, limit=10) == []


def test_empty_single_arm():
    result = rrf_pass1(
        {"dense": [], "sparse": [mk("a")]},
        {"dense": 60, "sparse": 60},
        limit=10,
    )

    assert len(result) == 1
    assert result[0].chunk_id == "a"


def test_all_arms_empty():
    result = rrf_pass1(
        {"dense": [], "sparse": []},
        {"dense": 60, "sparse": 60},
        limit=10,
    )

    assert result == []


def test_limit():
    hits = [mk(f"c{i}", case_id=i + 1) for i in range(50)]

    result = rrf_pass1({"dense": hits}, {"dense": 60}, limit=10)

    assert len(result) == 10


def test_ranks_are_one_indexed_and_contiguous():
    result = rrf_pass1(
        {"dense": [mk("a"), mk("b"), mk("c")]},
        {"dense": 60},
        limit=10,
    )

    assert [hit.rank for hit in result] == [1, 2, 3]


def test_duplicate_within_arm_counts_once():
    result = rrf_pass1(
        {"dense": [mk("a"), mk("a"), mk("b")]},
        {"dense": 60},
        limit=10,
    )

    assert result[0].score == pytest.approx(1 / 61)
    assert result[1].score == pytest.approx(1 / 62)


def test_duplicate_across_arms_accumulates():
    result = rrf_pass1(
        {"dense": [mk("a")], "sparse": [mk("a")]},
        {"dense": 60, "sparse": 60},
        limit=10,
    )

    assert len(result) == 1
    assert result[0].score == pytest.approx(2 / 61)


def test_parent_child_collision_raises():
    with pytest.raises(ValueError):
        rrf_pass1(
            {
                "dense": [mk("same", 1, False)],
                "sparse": [mk("same", 1, True)],
            },
            {"dense": 60, "sparse": 60},
            limit=10,
        )


def test_conflicting_case_id_raises():
    with pytest.raises(ValueError):
        rrf_pass1(
            {
                "dense": [mk("same", 1, False)],
                "sparse": [mk("same", 2, False)],
            },
            {"dense": 60, "sparse": 60},
            limit=10,
        )


def test_hit_not_mutated():
    original = mk("a")

    result = rrf_pass1({"dense": [original]}, {"dense": 60}, limit=10)

    assert result[0] is not original
    assert result[0].rank == 1
    assert original.rank == 0


def test_metadata_preserved_through_fusion():
    hit = Hit(
        chunk_id="a",
        case_id=1,
        is_parent=False,
        meta={"preview": "text", "court": "ca9", "hop": 1},
    )

    result = rrf_pass1({"dense": [hit]}, {"dense": 60}, limit=10)

    assert result[0].meta["court"] == "ca9"
    assert result[0].meta["hop"] == 1


def test_fusion_is_deterministic():
    arms = {"dense": [mk("a"), mk("b")], "sparse": [mk("b"), mk("c")]}
    ks = {"dense": 60, "sparse": 60}

    first = rrf_pass1(arms, ks, limit=10)
    second = rrf_pass1(arms, ks, limit=10)

    assert [h.chunk_id for h in first] == [h.chunk_id for h in second]
    assert [h.score for h in first] == [h.score for h in second]


def test_case_dedup_keeps_first_occurrence():
    hits = [mk("a", 1), mk("b", 1), mk("c", 2)]

    result = case_dedup(hits, max_per_case=1)

    assert [h.chunk_id for h in result] == ["a", "c"]


def test_case_dedup_allows_multiple_per_case():
    hits = [mk("a", 1), mk("b", 1), mk("c", 1), mk("d", 2)]

    result = case_dedup(hits, max_per_case=2)

    assert [h.chunk_id for h in result] == ["a", "b", "d"]


def test_case_dedup_preserves_order_and_ranks():
    hits = [
        Hit(chunk_id="a", case_id=1, is_parent=False, rank=1),
        Hit(chunk_id="b", case_id=1, is_parent=False, rank=2),
        Hit(chunk_id="c", case_id=2, is_parent=False, rank=3),
    ]

    result = case_dedup(hits, max_per_case=1)

    assert [h.rank for h in result] == [1, 3]


def test_case_dedup_empty():
    assert case_dedup([], max_per_case=1) == []


def test_case_dedup_zero_cap_raises():
    with pytest.raises(ValueError):
        case_dedup([mk("a")], max_per_case=0)
