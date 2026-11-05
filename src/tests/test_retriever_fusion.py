import pytest

from retriever.fusion import (
    Hit,
    rrf_pass1,
    rrf_pass2,
)


def mk(chunk_id, case_id=1, is_parent=False):
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
        {
            "dense": 60,
            "sparse": 60,
        },
    )

    assert result[0].chunk_id == "b"


def test_tie_broken_by_chunk_id():
    result = rrf_pass1(
        {
            "dense": [mk("a"), mk("b")],
            "sparse": [mk("b"), mk("a")],
        },
        {
            "dense": 60,
            "sparse": 60,
        },
    )

    assert result[0].chunk_id == "a"


def test_missing_k_raises():
    with pytest.raises(ValueError):
        rrf_pass1(
            {"dense": [mk("a")]},
            {"sparse": 60},
        )


def test_empty_arms():
    assert rrf_pass1({}, {}) == []


def test_empty_single_arm():
    result = rrf_pass1(
        {
            "dense": [],
            "sparse": [mk("a")],
        },
        {
            "dense": 60,
            "sparse": 60,
        },
    )

    assert len(result) == 1


def test_limit():
    hits = [mk(f"c{i}") for i in range(50)]

    result = rrf_pass1(
        {"dense": hits},
        {"dense": 60},
        limit=10,
    )

    assert len(result) == 10


def test_duplicate_within_arm_counts_once():
    result = rrf_pass1(
        {"dense": [mk("a"), mk("a"), mk("b")]},
        {"dense": 60},
    )

    assert result[0].score == pytest.approx(1 / 61)


def test_parent_child_collision_raises():
    with pytest.raises(ValueError):
        rrf_pass1(
            {
                "dense": [mk("same", 1, False)],
                "sparse": [mk("same", 1, True)],
            },
            {
                "dense": 60,
                "sparse": 60,
            },
        )


def test_pass2_seeds_dominate():
    result = rrf_pass2(
        [mk("a"), mk("b")],
        [mk("c", 3, True), mk("d", 4, True)],
    )

    assert result[0].chunk_id == "a"


def test_pass2_shared_result_boosted():
    result = rrf_pass2(
        [mk("a"), mk("b")],
        [mk("b", 1, False)],
    )

    assert result[0].chunk_id == "b"


def test_pass2_empty_graph_preserves_seeds():
    result = rrf_pass2(
        [mk("a"), mk("b")],
        [],
    )

    assert [hit.chunk_id for hit in result] == ["a", "b"]


def test_pass2_empty_inputs():
    assert rrf_pass2([], []) == []


def test_hit_not_mutated():
    original = mk("a")

    result = rrf_pass1(
        {"dense": [original]},
        {"dense": 60},
    )

    assert result[0] is not original
    assert result[0].rank == 1
    assert original.rank == 0
