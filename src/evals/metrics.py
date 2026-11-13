"""Deterministic metrics for legal retrieval evaluation.

Provides standard IR metrics (Recall, NDCG, MRR), bootstrap confidence
intervals for statistical rigor, and failure mode categorization to
diagnose exactly why queries fail.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from math import isfinite, log2
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class Prediction:
    """Normalized retrieval result for metric computation."""

    id: str
    retrieved_case_ids: list[int]
    retrieved_chunk_ids: list[str]
    query_class_actual: str | None
    arms_used: list[str]
    latency_ms: float
    empty: bool
    answerable: bool


class FailureMode(Enum):
    SUCCESS = "success"
    CLASSIFICATION_ERROR = "classification"
    RESOLUTION_ERROR = "resolution"
    RETRIEVAL_ERROR = "retrieval"
    RANKING_ERROR = "ranking"
    EMPTY_RESULTS = "empty"
    FALSE_POSITIVE = "false_positive"


def analyze_failure(
    prediction: Prediction,
    gold_case_ids: Sequence[int],
    gold_query_class: str,
    top_k: int = 10,
) -> FailureMode:
    """Categorize the outcome of one query.

    Negative controls (empty gold) succeed when the system returns
    nothing, and fail as FALSE_POSITIVE when it returns anything.

    Resolution failure is inferred from arms_used: the router only
    appends graph_1hop when the target case resolved, so its absence
    on a LOOKUP/VALIDITY query means resolution failed.
    """
    gold_set = set(gold_case_ids)

    if not gold_set:
        return FailureMode.SUCCESS if prediction.empty else FailureMode.FALSE_POSITIVE

    if prediction.query_class_actual != gold_query_class:
        return FailureMode.CLASSIFICATION_ERROR

    if gold_query_class in ("LOOKUP", "VALIDITY") and "graph_1hop" not in prediction.arms_used:
        return FailureMode.RESOLUTION_ERROR

    if prediction.empty:
        return FailureMode.EMPTY_RESULTS

    if not (gold_set & set(prediction.retrieved_case_ids)):
        return FailureMode.RETRIEVAL_ERROR

    if not (gold_set & set(prediction.retrieved_case_ids[:top_k])):
        return FailureMode.RANKING_ERROR

    return FailureMode.SUCCESS


def bootstrap_ci(
    values: Sequence[float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Compute mean and bootstrap confidence interval.

    Returns:
        (mean, lower_bound, upper_bound)
    """
    if not values:
        return 0.0, 0.0, 0.0

    arr = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed=42)
    samples = rng.choice(arr, size=(n_bootstrap, len(arr)), replace=True)
    means = samples.mean(axis=1)

    lower_pct = (1.0 - confidence) / 2.0 * 100.0
    upper_pct = (1.0 + confidence) / 2.0 * 100.0

    return (
        float(np.mean(arr)),
        float(np.percentile(means, lower_pct)),
        float(np.percentile(means, upper_pct)),
    )


def _validate_k(k: int) -> None:
    if not isinstance(k, int) or k < 1:
        raise ValueError(f"k must be positive int, got {k!r}")


def _unique(values: Sequence[Any]) -> list[Any]:
    return list(dict.fromkeys(values))


def recall_at_k(
    gold_ids: Sequence[int],
    retrieved_ids: Sequence[int],
    k: int,
) -> float:
    _validate_k(k)
    gold = set(gold_ids)
    if not gold:
        return 0.0
    retrieved = set(_unique(retrieved_ids)[:k])
    return len(gold & retrieved) / len(gold)


def ndcg_at_k(
    gold_ids: Sequence[int],
    retrieved_ids: Sequence[int],
    k: int,
) -> float:
    _validate_k(k)
    gold = set(gold_ids)
    if not gold:
        return 0.0

    retrieved = _unique(retrieved_ids)[:k]
    dcg = sum(
        1.0 / log2(rank + 1) for rank, case_id in enumerate(retrieved, start=1) if case_id in gold
    )

    ideal_count = min(len(gold), k)
    idcg = sum(1.0 / log2(rank + 1) for rank in range(1, ideal_count + 1))

    return dcg / idcg if idcg else 0.0


def mrr(gold_ids: Sequence[int], retrieved_ids: Sequence[int]) -> float:
    gold = set(gold_ids)
    if not gold:
        return 0.0
    for rank, case_id in enumerate(_unique(retrieved_ids), start=1):
        if case_id in gold:
            return 1.0 / rank
    return 0.0


def empty_result_rate(predictions: Sequence[Prediction]) -> float:
    if not predictions:
        return 0.0
    return sum(p.empty for p in predictions) / len(predictions)


def refusal_accuracy(predictions: Sequence[Prediction]) -> float:
    """Accuracy on negative controls (unanswerable queries)."""
    negatives = [p for p in predictions if not p.answerable]
    if not negatives:
        return 0.0
    return sum(p.empty for p in negatives) / len(negatives)


def latency_percentiles(latencies_ms: Sequence[float]) -> dict[str, float | None]:
    values = [float(v) for v in latencies_ms if isfinite(v) and v >= 0]
    if not values:
        return {"p50_ms": None, "p95_ms": None, "mean_ms": None}

    ordered = sorted(values)
    return {
        "p50_ms": float(np.percentile(ordered, 50)),
        "p95_ms": float(np.percentile(ordered, 95)),
        "mean_ms": sum(ordered) / len(ordered),
    }


__all__ = [
    "Prediction",
    "FailureMode",
    "analyze_failure",
    "bootstrap_ci",
    "recall_at_k",
    "ndcg_at_k",
    "mrr",
    "empty_result_rate",
    "refusal_accuracy",
    "latency_percentiles",
]
