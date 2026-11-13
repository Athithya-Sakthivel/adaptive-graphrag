"""Generic corpus-guarded evaluation runner.

Loads a verified gold set, executes the retriever adapter, computes
metrics with bootstrap confidence intervals, and categorizes failures.
Refuses to score if the live corpus fingerprint diverges from the
gold set's fingerprint (prevents silent chunker drift).
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from .metrics import (
    Prediction,
    analyze_failure,
    bootstrap_ci,
    empty_result_rate,
    latency_percentiles,
    mrr,
    ndcg_at_k,
    recall_at_k,
    refusal_accuracy,
)


class EvalRuntime(Protocol):
    """Structural interface every eval adapter must implement."""

    def corpus_fingerprint(self) -> str: ...
    def retrieve(self, query: str, *, top_k: int) -> dict[str, Any]: ...
    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class GoldRow:
    id: str
    suite: str
    query: str
    query_class_expected: str
    answerable: bool
    gold_case_ids: list[int]
    gold_chunk_ids: list[str]
    seed_case_id: int | None
    verified: bool
    corpus_hash: str


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    output = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: JSONL row must be an object")
            output.append(value)
    if not output:
        raise ValueError(f"Empty JSONL file: {path}")
    return output


def validate_gold(raw_rows: list[dict]) -> list[GoldRow]:
    output = []
    seen_ids = set()
    for index, raw in enumerate(raw_rows, start=1):
        row_id = raw.get("id")
        if not isinstance(row_id, str) or not row_id.strip():
            raise ValueError(f"Row {index}: invalid id")
        if row_id in seen_ids:
            raise ValueError(f"Duplicate gold ID: {row_id}")
        seen_ids.add(row_id)

        query = raw.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"{row_id}: query must be a non-empty string")

        query_class = raw.get("query_class_expected")
        if not isinstance(query_class, str) or not query_class.strip():
            raise ValueError(f"{row_id}: query_class_expected must be a non-empty string")

        answerable = raw.get("answerable")
        if not isinstance(answerable, bool):
            raise ValueError(f"{row_id}: answerable must be bool")

        corpus_hash = raw.get("corpus_hash")
        if not isinstance(corpus_hash, str) or len(corpus_hash) != 64:
            raise ValueError(f"{row_id}: invalid corpus SHA-256")

        gold_case_ids = raw.get("gold_case_ids", [])
        if not isinstance(gold_case_ids, list) or any(
            isinstance(x, bool) or not isinstance(x, int) for x in gold_case_ids
        ):
            raise ValueError(f"{row_id}: gold_case_ids must be a list of ints")

        gold_chunk_ids = raw.get("gold_chunk_ids", [])
        if not isinstance(gold_chunk_ids, list) or any(
            not isinstance(x, str) or not x for x in gold_chunk_ids
        ):
            raise ValueError(f"{row_id}: gold_chunk_ids must be a list of non-empty strings")

        if answerable and not (gold_case_ids or gold_chunk_ids):
            raise ValueError(f"{row_id}: answerable row has no gold evidence")
        if not answerable and (gold_case_ids or gold_chunk_ids):
            raise ValueError(f"{row_id}: negative control has non-empty gold")

        seed_id = raw.get("seed_case_id")
        if seed_id is not None and (isinstance(seed_id, bool) or not isinstance(seed_id, int)):
            raise ValueError(f"{row_id}: invalid seed_case_id")

        output.append(
            GoldRow(
                id=row_id,
                suite=raw.get("suite", "custom"),
                query=query,
                query_class_expected=query_class,
                answerable=answerable,
                gold_case_ids=gold_case_ids,
                gold_chunk_ids=gold_chunk_ids,
                seed_case_id=seed_id,
                verified=bool(raw.get("verified")),
                corpus_hash=corpus_hash.lower(),
            )
        )
    return output


def import_factory(spec: str) -> Any:
    if ":" not in spec:
        raise ValueError("Adapter must use module:function syntax")
    module_name, function_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, function_name, None)
    if not callable(factory):
        raise TypeError(f"Adapter is not callable: {spec}")
    return factory


def normalise_prediction(gold: GoldRow, result: dict, elapsed_ms: float) -> Prediction:
    hits = result.get("hits") or []
    case_ids = []
    chunk_ids = []

    for hit in hits:
        if "case_id" in hit:
            case_ids.append(hit["case_id"])
        if "chunk_id" in hit:
            chunk_ids.append(hit["chunk_id"])

    return Prediction(
        id=gold.id,
        retrieved_case_ids=list(dict.fromkeys(case_ids)),
        retrieved_chunk_ids=list(dict.fromkeys(chunk_ids)),
        query_class_actual=result.get("query_class_actual"),
        arms_used=result.get("arms_used") or [],
        latency_ms=elapsed_ms,
        empty=not (case_ids or chunk_ids),
        answerable=gold.answerable,
    )


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def score_slice(
    gold_rows: list[GoldRow],
    predictions: list[Prediction],
    top_k: int,
) -> dict:
    pairs = list(zip(gold_rows, predictions, strict=True))
    positive = [p for p in pairs if p[0].answerable]
    case_pairs = [p for p in positive if p[0].gold_case_ids]
    preds = [p for _, p in pairs]

    # Calculate for both @5 and @top_k
    case_recalls_5 = [recall_at_k(g.gold_case_ids, p.retrieved_case_ids, 5) for g, p in case_pairs]
    case_recalls_k = [
        recall_at_k(g.gold_case_ids, p.retrieved_case_ids, top_k) for g, p in case_pairs
    ]
    case_ndcgs = [ndcg_at_k(g.gold_case_ids, p.retrieved_case_ids, top_k) for g, p in case_pairs]
    case_mrrs = [
        mrr(g.gold_case_ids, p.retrieved_case_ids) for g, p in case_pairs
    ]  # <-- ADD THIS LINE BACK

    def ci_or_none(values: list[float]) -> tuple[float | None, list[float] | None]:
        if not values:
            return None, None
        mean, lo, hi = bootstrap_ci(values)
        return mean, [lo, hi]

    rec_5_mean, rec_5_ci = ci_or_none(case_recalls_5)
    rec_k_mean, rec_k_ci = ci_or_none(case_recalls_k)
    ndcg_mean, ndcg_ci = ci_or_none(case_ndcgs)
    mrr_mean, mrr_ci = ci_or_none(case_mrrs)

    return {
        "n": len(pairs),
        "n_answerable": len(positive),
        "n_case_scored": len(case_pairs),
        "case_recall@5": rec_5_mean,
        "case_recall@5_ci": rec_5_ci,
        f"case_recall@{top_k}": rec_k_mean,
        f"case_recall@{top_k}_ci": rec_k_ci,
        f"case_ndcg@{top_k}": ndcg_mean,
        f"case_ndcg@{top_k}_ci": ndcg_ci,
        "case_mrr": mrr_mean,
        "case_mrr_ci": mrr_ci,
        "empty_result_rate": empty_result_rate(preds),
        "refusal_accuracy": refusal_accuracy(preds),
        "latency": latency_percentiles([p.latency_ms for _, p in pairs]),
    }


def evaluate(
    *,
    gold_path: Path,
    out_path: Path,
    adapter_spec: str,
    config_name: str,
    suite: str,
    top_k: int,
    exact_baseline: bool,
    published_only: bool = False,
    no_graph: bool,
    allow_unverified: bool,
    rrf_k: int | None,
) -> dict:
    if top_k < 1:
        raise ValueError("top_k must be positive")

    raw_rows = load_jsonl(gold_path)
    gold_rows = validate_gold(raw_rows)

    suites = {row.suite for row in gold_rows}
    if suites != {suite}:
        raise ValueError(f"Gold suite mismatch: expected {suite!r}; found {sorted(suites)}")

    unverified = [row.id for row in gold_rows if not row.verified]
    if unverified and not allow_unverified:
        raise PermissionError(
            f"Refusing {len(unverified)} unverified gold entries. "
            f"Pass --allow-unverified to override."
        )

    corpus_hashes = {row.corpus_hash for row in gold_rows}
    if len(corpus_hashes) != 1:
        raise ValueError("Gold contains multiple corpus fingerprints")
    expected_hash = next(iter(corpus_hashes))

    factory = import_factory(adapter_spec)
    runtime = cast(
        EvalRuntime,
        factory(
            suite=suite,
            config_name=config_name,
            exact_baseline=exact_baseline,
            published_only=published_only,
            no_graph=no_graph,
            rrf_k=rrf_k,
            overrides={},
        ),
    )

    predictions = []
    try:
        actual_hash = runtime.corpus_fingerprint()
        if actual_hash.lower() != expected_hash:
            raise RuntimeError(
                f"CORPUS HASH MISMATCH. Gold={expected_hash[:16]}...; "
                f"Actual={actual_hash[:16]}.... Refusing to score."
            )

        for index, gold in enumerate(gold_rows, start=1):
            started = time.perf_counter()
            result = runtime.retrieve(gold.query, top_k=top_k)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            predictions.append(normalise_prediction(gold, result, elapsed_ms))

            if index % 25 == 0 or index == len(gold_rows):
                print(f"Evaluated {index}/{len(gold_rows)}", file=sys.stderr)
    finally:
        if hasattr(runtime, "close"):
            runtime.close()

    grouped: dict[str, list[tuple[GoldRow, Prediction]]] = {}
    for gold, prediction in zip(gold_rows, predictions, strict=True):
        label = gold.query_class_expected if gold.answerable else "negative_control"
        grouped.setdefault(label, []).append((gold, prediction))

    per_class = {}
    failure_modes = {}

    for label, pairs in sorted(grouped.items()):
        per_class[label] = score_slice(
            [g for g, _ in pairs],
            [p for _, p in pairs],
            top_k,
        )

        counts: dict[str, int] = {}
        for g, p in pairs:
            mode = analyze_failure(p, g.gold_case_ids, g.query_class_expected, top_k).value
            counts[mode] = counts.get(mode, 0) + 1
        failure_modes[label] = dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True))

    positive_labels = [label for label in per_class if label != "negative_control"]

    def macro(key: str) -> float | None:
        values = [
            per_class[label].get(key)
            for label in positive_labels
            if per_class[label].get(key) is not None
        ]
        return sum(values) / len(values) if values else None

    report = {
        "schema_version": 1,
        "suite": suite,
        "config": config_name,
        "top_k": top_k,
        "exact_baseline": exact_baseline,
        "no_graph": no_graph,
        "rrf_k": rrf_k,
        "n_queries": len(gold_rows),
        "corpus_hash": expected_hash,
        "per_class": per_class,
        "failure_modes": failure_modes,
        "macro_positive": {
            f"case_recall@{top_k}": macro(f"case_recall@{top_k}"),
            f"case_ndcg@{top_k}": macro(f"case_ndcg@{top_k}"),
        },
        "latency": latency_percentiles([p.latency_ms for p in predictions]),
    }

    atomic_write_json(out_path, report)
    print(json.dumps(report, indent=2))
    print(f"Metrics: {out_path}")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run retrieval evaluation against verified gold.")
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--suite", default="custom")
    parser.add_argument("--config", default="baseline")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--exact-baseline", action="store_true")
    parser.add_argument("--published-only", action="store_true")
    parser.add_argument("--no-graph", action="store_true")
    parser.add_argument("--allow-unverified", action="store_true")
    parser.add_argument("--rrf-k", type=int)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        evaluate(
            gold_path=args.gold,
            out_path=args.out,
            adapter_spec=args.adapter,
            config_name=args.config,
            suite=args.suite,
            top_k=args.top_k,
            exact_baseline=args.exact_baseline,
            published_only=args.published_only,
            no_graph=args.no_graph,
            allow_unverified=args.allow_unverified,
            rrf_k=args.rrf_k,
        )
        return 0
    except Exception as exc:
        print(f"EVALUATION FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
