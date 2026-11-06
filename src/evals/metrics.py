"""Deterministic metrics for legal retrieval evaluation.

PEP 695 type aliases (``type Span = ...``) are used throughout; this
requires Python 3.12+.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite, log2
from typing import Any

type Span = tuple[str, int, int]


@dataclass(frozen=True, slots=True)
class Prediction:
    id: str
    retrieved_case_ids: list[int]
    retrieved_chunk_ids: list[str]
    retrieved_spans: list[Span]
    query_class_actual: str | None
    arms_used: list[str]
    arm_diagnostics: list[dict[str, Any]]
    latency_ms: float
    empty: bool
    answerable: bool


def _unique(values: Sequence[Any]) -> list[Any]:
    return list(dict.fromkeys(values))


def _validate_k(k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")


def case_recall_at_k(
    gold_case_ids: Sequence[int],
    retrieved_case_ids: Sequence[int],
    k: int,
) -> float:
    _validate_k(k)
    gold = set(gold_case_ids)
    if not gold:
        return 0.0
    retrieved = set(_unique(retrieved_case_ids)[:k])
    return len(gold & retrieved) / len(gold)


def chunk_recall_at_k(
    gold_chunk_ids: Sequence[str],
    retrieved_chunk_ids: Sequence[str],
    k: int,
) -> float:
    _validate_k(k)
    gold = set(gold_chunk_ids)
    if not gold:
        return 0.0
    retrieved = set(_unique(retrieved_chunk_ids)[:k])
    return len(gold & retrieved) / len(gold)


def case_ndcg_at_k(
    gold_case_ids: Sequence[int],
    retrieved_case_ids: Sequence[int],
    k: int,
) -> float:
    _validate_k(k)
    gold = set(gold_case_ids)
    if not gold:
        return 0.0
    retrieved = _unique(retrieved_case_ids)[:k]
    dcg = sum(
        1.0 / log2(rank + 1) for rank, case_id in enumerate(retrieved, start=1) if case_id in gold
    )
    ideal_count = min(len(gold), k)
    idcg = sum(1.0 / log2(rank + 1) for rank in range(1, ideal_count + 1))
    return dcg / idcg if idcg else 0.0


def mrr(
    gold_case_ids: Sequence[int],
    retrieved_case_ids: Sequence[int],
) -> float:
    gold = set(gold_case_ids)
    if not gold:
        return 0.0
    for rank, case_id in enumerate(_unique(retrieved_case_ids), start=1):
        if case_id in gold:
            return 1.0 / rank
    return 0.0


def _coerce_span(value: Any) -> Span:
    if isinstance(value, Mapping):
        doc = value.get("doc", value.get("file_path"))
        start = value.get("start")
        end = value.get("end")
        if start is None or end is None:
            raw = value.get("span")
            if isinstance(raw, (list, tuple)) and len(raw) == 2:
                start, end = raw
    elif isinstance(value, (list, tuple)) and len(value) == 3:
        doc, start, end = value
    else:
        raise ValueError(f"Invalid span: {value!r}")

    if not isinstance(doc, str) or not doc:
        raise ValueError("Span document must be a non-empty string")

    if (
        isinstance(start, bool)
        or isinstance(end, bool)
        or not isinstance(start, int)
        or not isinstance(end, int)
        or start < 0
        or end <= start
    ):
        raise ValueError(f"Invalid half-open interval: {value!r}")

    return doc, start, end


def _merge_intervals(
    spans: Sequence[Any],
) -> dict[str, list[tuple[int, int]]]:
    grouped: dict[str, list[tuple[int, int]]] = {}
    for raw in spans:
        doc, start, end = _coerce_span(raw)
        grouped.setdefault(doc, []).append((start, end))

    merged: dict[str, list[tuple[int, int]]] = {}
    for doc, intervals in grouped.items():
        intervals.sort()
        output: list[tuple[int, int]] = []
        for start, end in intervals:
            if output and start <= output[-1][1]:
                old_start, old_end = output[-1]
                output[-1] = (old_start, max(old_end, end))
            else:
                output.append((start, end))
        merged[doc] = output
    return merged


def _total_length(
    intervals: Mapping[str, Sequence[tuple[int, int]]],
) -> int:
    return sum(end - start for values in intervals.values() for start, end in values)


def _intersection_length(
    left: Mapping[str, Sequence[tuple[int, int]]],
    right: Mapping[str, Sequence[tuple[int, int]]],
) -> int:
    total = 0
    for doc in left.keys() & right.keys():
        a = left[doc]
        b = right[doc]
        i = 0
        j = 0
        while i < len(a) and j < len(b):
            start = max(a[i][0], b[j][0])
            end = min(a[i][1], b[j][1])
            if end > start:
                total += end - start
            if a[i][1] <= b[j][1]:
                i += 1
            else:
                j += 1
    return total


def char_span_recall(
    gold_spans: Sequence[Any],
    retrieved_spans: Sequence[Any],
) -> float:
    gold = _merge_intervals(gold_spans)
    denominator = _total_length(gold)
    if denominator == 0:
        return 0.0
    retrieved = _merge_intervals(retrieved_spans)
    return _intersection_length(gold, retrieved) / denominator


def char_span_precision(
    gold_spans: Sequence[Any],
    retrieved_spans: Sequence[Any],
) -> float:
    retrieved = _merge_intervals(retrieved_spans)
    denominator = _total_length(retrieved)
    if denominator == 0:
        return 0.0
    gold = _merge_intervals(gold_spans)
    return _intersection_length(gold, retrieved) / denominator


def _value(row: Any, field: str, default: Any = None) -> Any:
    if isinstance(row, Mapping):
        return row.get(field, default)
    return getattr(row, field, default)


def _has_hits(row: Any) -> bool:
    for field in (
        "retrieved_case_ids",
        "retrieved_chunk_ids",
        "retrieved_spans",
    ):
        if _value(row, field, []):
            return True
    return False


def empty_result_rate(rows: Sequence[Any]) -> float:
    if not rows:
        return 0.0
    return sum(not _has_hits(row) for row in rows) / len(rows)


def refusal_accuracy(rows: Sequence[Any]) -> float:
    negatives = [row for row in rows if _value(row, "answerable", True) is False]
    if not negatives:
        return 0.0
    return sum(not _has_hits(row) for row in negatives) / len(negatives)


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("No observations")
    position = (len(values) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def latency_percentiles(
    latencies_ms: Sequence[float],
) -> dict[str, float | None]:
    values = []
    for value in latencies_ms:
        number = float(value)
        if not isfinite(number) or number < 0:
            raise ValueError(f"Invalid latency: {value!r}")
        values.append(number)
    if not values:
        return {
            "p50_ms": None,
            "p95_ms": None,
            "mean_ms": None,
            "min_ms": None,
            "max_ms": None,
        }
    ordered = sorted(values)
    return {
        "p50_ms": _percentile(ordered, 0.50),
        "p95_ms": _percentile(ordered, 0.95),
        "mean_ms": sum(ordered) / len(ordered),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }
