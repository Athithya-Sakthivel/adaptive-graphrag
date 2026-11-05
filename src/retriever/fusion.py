"""Deterministic reciprocal-rank fusion with per-case deduplication.

RRF uses rank rather than comparing incompatible raw scores from
cosine distance, PostgreSQL lexical ranking, and graph retrieval.

Dense and sparse arms return child chunks. Exact and graph arms
return parent chunks. A child and its parent are distinct retrieval
units; parent expansion is a later context-building operation, not
an RRF deduplication rule.

`case_dedup` enforces the per-query-class case diversity policy by
keeping at most `max_per_case` hits per case ID while preserving
rank order. Per-class caps live in `config.case_max_per_class`;
this module only applies the cap it is given.

`rrf_pass1` and `rrf_pass2` accept an optional `RetrieverConfig`.
When `config` is supplied without an explicit `k_per_arm` (or
explicit pass-2 weights), the config is the source of truth. When
explicit values are supplied, they always win — the config is only
consulted for parameters the caller did not pass.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config import RetrieverConfig


@dataclass(frozen=True, slots=True)
class Hit:
    """One retrieval result.

    score:
        Raw arm score or fused RRF score. Its meaning depends on the arm.
        Dense scores are cosine distances (lower is better); sparse scores
        are ts_rank_cd values (higher is better). Fusion uses rank only.

    rank:
        1-indexed position within an arm or final fused result.
        Zero means that fusion has not yet assigned a rank.

    meta:
        Always includes preview. Other metadata fields are optional.
    """

    chunk_id: str
    case_id: int
    is_parent: bool
    score: float = 0.0
    rank: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.chunk_id, str) or not self.chunk_id.strip():
            raise ValueError("chunk_id must be a non-empty string")

        if isinstance(self.case_id, bool) or not isinstance(self.case_id, int):
            raise TypeError("case_id must be an integer")

        if self.case_id <= 0:
            raise ValueError("case_id must be positive")

        if not isinstance(self.is_parent, bool):
            raise TypeError("is_parent must be bool")

        if isinstance(self.score, bool) or not isinstance(self.score, (int, float)):
            raise TypeError("score must be numeric")

        if not math.isfinite(float(self.score)):
            raise ValueError("score must be finite")

        if isinstance(self.rank, bool) or not isinstance(self.rank, int):
            raise TypeError("rank must be an integer")

        if self.rank < 0:
            raise ValueError("rank cannot be negative")

        if not isinstance(self.meta, Mapping):
            raise TypeError("meta must be a mapping")

        # Copy metadata so callers cannot mutate the original dictionary
        # after constructing a Hit. The record remains frozen at the
        # dataclass level.
        metadata = dict(self.meta)
        preview = metadata.get("preview", "")

        if preview is None:
            preview = ""

        if not isinstance(preview, str):
            preview = str(preview)

        metadata["preview"] = preview
        object.__setattr__(self, "meta", metadata)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_limit(limit: int) -> None:
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be a positive integer")


def _validate_max_per_case(max_per_case: int) -> None:
    if isinstance(max_per_case, bool) or not isinstance(max_per_case, int) or max_per_case < 1:
        raise ValueError("max_per_case must be a positive integer")


def _validate_k(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _validate_weights(weight_seeds: float, weight_graph: float) -> None:
    for name, value in (
        ("weight_seeds", weight_seeds),
        ("weight_graph", weight_graph),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric")

        if not math.isfinite(float(value)) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")

    if weight_seeds + weight_graph <= 0:
        raise ValueError("at least one fusion weight must be positive")


def _validate_hit_list(hits: list[Hit], arm_name: str) -> None:
    if not isinstance(hits, list):
        raise TypeError(f"{arm_name} results must be a list")

    for hit in hits:
        if not isinstance(hit, Hit):
            raise TypeError(f"{arm_name} contains {type(hit).__name__}, expected Hit")


def _register_hit(
    hit: Hit,
    *,
    metadata: dict[str, dict[str, Any]],
    case_ids: dict[str, int],
    parents: dict[str, bool],
) -> None:
    """Register chunk identity and reject contradictory records."""

    chunk_id = hit.chunk_id

    if chunk_id not in metadata:
        metadata[chunk_id] = dict(hit.meta)
        case_ids[chunk_id] = hit.case_id
        parents[chunk_id] = hit.is_parent
        return

    if case_ids[chunk_id] != hit.case_id:
        raise ValueError(f"chunk {chunk_id!r} appears with conflicting case IDs")

    if parents[chunk_id] != hit.is_parent:
        raise ValueError(f"chunk {chunk_id!r} appears as both parent and child")


# ---------------------------------------------------------------------------
# Case-level deduplication
# ---------------------------------------------------------------------------


def case_dedup(hits: list[Hit], max_per_case: int = 1) -> list[Hit]:
    """Keep at most ``max_per_case`` hits per ``case_id``.

    Order is preserved: the first ``max_per_case`` occurrences of each
    case ID are kept, later occurrences are dropped. Hit ``rank`` and
    ``score`` fields are left untouched so that callers can re-rank or
    re-score after dedup if the downstream stage requires it.

    This is the enforcement point for the per-query-class case
    diversity policy. The caller (router) chooses ``max_per_case``
    from ``config.case_max_per_class(query_class)``; this function
    does not know about query classes.
    """

    _validate_hit_list(hits, "hits")
    _validate_max_per_case(max_per_case)

    counts: dict[int, int] = {}
    output: list[Hit] = []

    for hit in hits:
        if counts.get(hit.case_id, 0) >= max_per_case:
            continue
        counts[hit.case_id] = counts.get(hit.case_id, 0) + 1
        output.append(hit)

    return output


# ---------------------------------------------------------------------------
# Config-driven parameter resolution
# ---------------------------------------------------------------------------


def _resolve_k_per_arm(
    arms: Mapping[str, list[Hit]],
    k_per_arm: Mapping[str, int] | None,
    config: RetrieverConfig | None,
) -> Mapping[str, int]:
    """Resolve per-arm k values.

    Precedence:
        1. Explicit ``k_per_arm`` — always wins, no merge with config.
        2. ``config`` — ``getattr(config, f"{arm}_k")`` per arm.
        3. ``default_config()`` — same lookup as (2).
    """

    if k_per_arm is not None:
        return k_per_arm

    if config is None:
        from .config import default_config

        config = default_config()

    resolved: dict[str, int] = {}

    for arm_name in arms:
        attr = f"{arm_name}_k"
        value = getattr(config, attr, None)

        if value is None:
            raise ValueError(
                f"config has no attribute {attr!r}; pass k_per_arm explicitly for arm {arm_name!r}"
            )

        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"config.{attr} must be an integer")

        resolved[arm_name] = value

    return resolved


_DEFAULT_PASS2_KWARGS: dict[str, float | int] = {
    "weight_seeds": 0.85,
    "weight_graph": 0.15,
    "k_seeds": 60,
    "k_graph": 10,
}


def _resolve_pass2_kwargs(
    config: RetrieverConfig | None,
    weight_seeds: float | None,
    weight_graph: float | None,
    k_seeds: int | None,
    k_graph: int | None,
) -> dict[str, float | int]:
    """Resolve pass-2 fusion parameters.

    Precedence per parameter:
        1. Explicit keyword argument.
        2. ``config.application_pass2_kwargs()`` if config is provided.
        3. Module-level defaults (preserves pre-config behavior).
    """

    resolved: dict[str, float | int] = dict(_DEFAULT_PASS2_KWARGS)

    if config is not None:
        method = getattr(config, "application_pass2_kwargs", None)
        if callable(method):
            overrides = method()
            if isinstance(overrides, Mapping):
                for key in resolved:
                    if key in overrides:
                        resolved[key] = overrides[key]

    if weight_seeds is not None:
        resolved["weight_seeds"] = weight_seeds
    if weight_graph is not None:
        resolved["weight_graph"] = weight_graph
    if k_seeds is not None:
        resolved["k_seeds"] = k_seeds
    if k_graph is not None:
        resolved["k_graph"] = k_graph

    return resolved


# ---------------------------------------------------------------------------
# Pass 1: multi-arm RRF
# ---------------------------------------------------------------------------


def rrf_pass1(
    arms: dict[str, list[Hit]],
    k_per_arm: Mapping[str, int] | None = None,
    limit: int = 30,
    *,
    config: RetrieverConfig | None = None,
) -> list[Hit]:
    """Fuse one or more ranked retrieval arms.

    Formula:
        RRF(chunk) = sum(1 / (k_arm + one_based_rank))

    Duplicate chunk IDs within one arm count only once, using their
    first occurrence. Duplicates across different arms accumulate
    contributions.

    Equal fused scores are resolved by chunk_id ascending.
    """

    _validate_limit(limit)

    if not isinstance(arms, Mapping):
        raise TypeError("arms must be a mapping")

    if not arms:
        return []

    effective_k = _resolve_k_per_arm(arms, k_per_arm, config)

    missing = set(arms) - set(effective_k)

    if missing:
        raise ValueError(f"arms missing k values: {sorted(missing)}")

    scores: dict[str, float] = {}
    metadata: dict[str, dict[str, Any]] = {}
    case_ids: dict[str, int] = {}
    parents: dict[str, bool] = {}

    for arm_name, hits in arms.items():
        if not isinstance(arm_name, str) or not arm_name:
            raise ValueError("arm names must be non-empty strings")

        k = effective_k[arm_name]
        _validate_k(k, f"k for arm {arm_name!r}")
        _validate_hit_list(hits, arm_name)

        seen: set[str] = set()

        for hit in hits:
            # A malformed arm must not give one chunk multiple votes
            # simply because it accidentally emitted duplicate rows.
            if hit.chunk_id in seen:
                continue

            seen.add(hit.chunk_id)

            _register_hit(
                hit,
                metadata=metadata,
                case_ids=case_ids,
                parents=parents,
            )

            rank = len(seen)
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (k + rank)

    ordered = sorted(
        scores,
        key=lambda chunk_id: (-scores[chunk_id], chunk_id),
    )[:limit]

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


# ---------------------------------------------------------------------------
# Pass 2: seed+graph fusion
# ---------------------------------------------------------------------------


def rrf_pass2(
    seeds: list[Hit],
    graph_hits: list[Hit],
    *,
    weight_seeds: float | None = None,
    weight_graph: float | None = None,
    k_seeds: int | None = None,
    k_graph: int | None = None,
    limit: int = 20,
    config: RetrieverConfig | None = None,
) -> list[Hit]:
    """Merge already-fused seed hits with graph-derived parent hits.

    Formula:
        seed contribution  = weight_seeds / (k_seeds + rank)
        graph contribution = weight_graph / (k_graph + rank)

    If the graph arm is empty, the seed order is retained. This is
    deliberate graceful degradation: a missing graph path must not
    destroy valid first-pass retrieval.

    All keyword parameters are optional. When omitted, values are
    resolved in this order:
        1. Explicit keyword argument.
        2. ``config.application_pass2_kwargs()``.
        3. Module defaults (0.85 / 0.15 / 60 / 10).
    """

    _validate_limit(limit)

    resolved = _resolve_pass2_kwargs(config, weight_seeds, weight_graph, k_seeds, k_graph)

    r_weight_seeds = float(resolved["weight_seeds"])
    r_weight_graph = float(resolved["weight_graph"])
    r_k_seeds = int(resolved["k_seeds"])
    r_k_graph = int(resolved["k_graph"])

    _validate_k(r_k_seeds, "k_seeds")
    _validate_k(r_k_graph, "k_graph")
    _validate_weights(r_weight_seeds, r_weight_graph)

    _validate_hit_list(seeds, "seeds")
    _validate_hit_list(graph_hits, "graph_hits")

    if not seeds and not graph_hits:
        return []

    if not graph_hits:
        # Preserve the first-pass scores when graph retrieval contributes
        # nothing. Reassign ranks defensively without mutating inputs.
        return [
            Hit(
                chunk_id=hit.chunk_id,
                case_id=hit.case_id,
                is_parent=hit.is_parent,
                score=hit.score,
                rank=index,
                meta=hit.meta,
            )
            for index, hit in enumerate(seeds[:limit], start=1)
        ]

    scores: dict[str, float] = {}
    metadata: dict[str, dict[str, Any]] = {}
    case_ids: dict[str, int] = {}
    parents: dict[str, bool] = {}

    def accumulate(
        hits: list[Hit],
        *,
        weight: float,
        k: int,
        arm_name: str,
    ) -> None:
        seen: set[str] = set()

        for hit in hits:
            if hit.chunk_id in seen:
                continue

            seen.add(hit.chunk_id)

            _register_hit(
                hit,
                metadata=metadata,
                case_ids=case_ids,
                parents=parents,
            )

            rank = len(seen)
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + weight / (k + rank)

    accumulate(
        seeds,
        weight=r_weight_seeds,
        k=r_k_seeds,
        arm_name="seeds",
    )

    accumulate(
        graph_hits,
        weight=r_weight_graph,
        k=r_k_graph,
        arm_name="graph",
    )

    ordered = sorted(
        scores,
        key=lambda chunk_id: (-scores[chunk_id], chunk_id),
    )[:limit]

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
    "rrf_pass2",
]
