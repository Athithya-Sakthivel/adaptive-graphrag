"""Deterministic reciprocal-rank fusion with per-case deduplication.

RRF uses rank rather than comparing incompatible raw scores from
cosine distance, PostgreSQL lexical ranking, and graph traversal.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Hit:
    """One retrieval result.

    score:
        Raw arm score or fused RRF score. Meaning depends on arm.
        Dense: cosine distance (lower=better). Sparse: ts_rank_cd
        (higher=better). Fusion uses rank only.

    rank:
        1-indexed position within an arm or final fused result.
        Zero means fusion has not yet assigned a rank.

    meta:
        Always includes preview. Other fields are optional.
    """

    chunk_id: str
    case_id: int
    is_parent: bool
    score: float = 0.0
    rank: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.chunk_id, str) or not self.chunk_id.strip():
            raise ValueError("chunk_id must be non-empty str")
        if not isinstance(self.case_id, int) or self.case_id <= 0:
            raise ValueError(f"case_id must be positive int, got {self.case_id!r}")
        if not isinstance(self.is_parent, bool):
            raise TypeError("is_parent must be bool")
        if not isinstance(self.score, (int, float)):
            raise TypeError("score must be numeric")
        if not isinstance(self.rank, int) or self.rank < 0:
            raise ValueError("rank must be non-negative int")

        metadata = dict(self.meta)
        preview = metadata.get("preview", "")
        if preview is None:
            preview = ""
        if not isinstance(preview, str):
            preview = str(preview)
        metadata["preview"] = preview
        object.__setattr__(self, "meta", metadata)


def case_dedup(hits: list[Hit], max_per_case: int = 1) -> list[Hit]:
    """Keep at most max_per_case hits per case_id, preserving order.

    Rank and score fields are untouched so callers can re-rank after
    dedup if needed. This is the enforcement point for per-query-class
    case diversity policy.
    """
    if max_per_case < 1:
        raise ValueError(f"max_per_case must be positive, got {max_per_case}")

    counts: dict[int, int] = {}
    output: list[Hit] = []
    for hit in hits:
        current = counts.get(hit.case_id, 0)
        if current >= max_per_case:
            continue
        counts[hit.case_id] = current + 1
        output.append(hit)
    return output


def rrf_pass1(
    arms: Mapping[str, list[Hit]],
    k_per_arm: Mapping[str, int],
    limit: int,
) -> list[Hit]:
    """Fuse one or more ranked retrieval arms.

    Formula: RRF(chunk) = sum(1 / (k_arm + one_based_rank))

    Duplicate chunk IDs within one arm count only once (using their
    first occurrence). Duplicates across different arms accumulate
    contributions. Equal fused scores are resolved by chunk_id asc.
    """
    if limit < 1:
        raise ValueError(f"limit must be positive, got {limit}")
    if not arms:
        return []

    missing = set(arms) - set(k_per_arm)
    if missing:
        raise ValueError(f"arms missing k values: {sorted(missing)}")

    scores: dict[str, float] = {}
    metadata: dict[str, dict[str, Any]] = {}
    case_ids: dict[str, int] = {}
    parents: dict[str, bool] = {}

    for arm_name, hits in arms.items():
        k = k_per_arm[arm_name]
        if k < 1:
            raise ValueError(f"k for arm {arm_name!r} must be positive")

        seen: set[str] = set()
        for hit in hits:
            if hit.chunk_id in seen:
                continue
            seen.add(hit.chunk_id)

            if hit.chunk_id not in metadata:
                metadata[hit.chunk_id] = dict(hit.meta)
                case_ids[hit.chunk_id] = hit.case_id
                parents[hit.chunk_id] = hit.is_parent
            elif case_ids[hit.chunk_id] != hit.case_id:
                raise ValueError(f"chunk {hit.chunk_id!r} has conflicting case IDs")
            elif parents[hit.chunk_id] != hit.is_parent:
                raise ValueError(f"chunk {hit.chunk_id!r} appears as both parent and child")

            rank = len(seen)
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (k + rank)

    ordered = sorted(scores, key=lambda cid: (-scores[cid], cid))[:limit]

    return [
        Hit(
            chunk_id=chunk_id,
            case_id=case_ids[chunk_id],
            is_parent=parents[chunk_id],
            score=scores[chunk_id],
            rank=index,
            meta=metadata[chunk_id],
        )
        for index, chunk_id in enumerate(ordered, start=1)
    ]


__all__ = [
    "Hit",
    "case_dedup",
    "rrf_pass1",
]
