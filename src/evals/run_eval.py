"""Generic corpus-guarded evaluation runner."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from evals.metrics import (
    Prediction,
    case_ndcg_at_k,
    case_recall_at_k,
    char_span_precision,
    char_span_recall,
    chunk_recall_at_k,
    empty_result_rate,
    latency_percentiles,
    mrr,
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
    gold_char_spans: list[dict] | None
    seed_case_id: int | None
    tags: list[str]
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

        suite = raw.get("suite", "custom")
        query = raw.get("query")
        query_class = raw.get("query_class_expected", "UNKNOWN")
        answerable = raw.get("answerable")
        verified = raw.get("verified")
        corpus_hash = raw.get("corpus_hash")

        if not isinstance(suite, str) or not suite:
            raise ValueError(f"{row_id}: invalid suite")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"{row_id}: invalid query")
        if not isinstance(query_class, str) or not query_class:
            raise ValueError(f"{row_id}: invalid query class")
        if not isinstance(answerable, bool):
            raise ValueError(f"{row_id}: answerable must be boolean")
        if not isinstance(verified, bool):
            raise ValueError(f"{row_id}: verified must be boolean")
        if (
            not isinstance(corpus_hash, str)
            or len(corpus_hash) != 64
            or any(c not in "0123456789abcdefABCDEF" for c in corpus_hash)
        ):
            raise ValueError(f"{row_id}: invalid corpus SHA-256")

        case_ids = raw.get("gold_case_ids", [])
        chunk_ids = raw.get("gold_chunk_ids", [])
        spans = raw.get("gold_char_spans")

        if not isinstance(case_ids, list):
            raise ValueError(f"{row_id}: gold_case_ids must be a list")
        if any(isinstance(x, bool) or not isinstance(x, int) for x in case_ids):
            raise ValueError(f"{row_id}: invalid gold case ID")
        if not isinstance(chunk_ids, list) or any(
            not isinstance(x, str) or not x for x in chunk_ids
        ):
            raise ValueError(f"{row_id}: invalid gold chunk IDs")

        if spans is not None:
            if not isinstance(spans, list):
                raise ValueError(f"{row_id}: gold_char_spans must be a list or null")
            for span in spans:
                if not isinstance(span, dict):
                    raise ValueError(f"{row_id}: span must be an object")
                doc = span.get("doc")
                start = span.get("start")
                end = span.get("end")
                if (
                    not isinstance(doc, str)
                    or not doc
                    or isinstance(start, bool)
                    or isinstance(end, bool)
                    or not isinstance(start, int)
                    or not isinstance(end, int)
                    or start < 0
                    or end <= start
                ):
                    raise ValueError(f"{row_id}: invalid gold span")

        has_gold = bool(case_ids or chunk_ids or spans)
        if answerable and not has_gold:
            raise ValueError(f"{row_id}: answerable entry has no gold evidence")
        if not answerable and has_gold:
            raise ValueError(f"{row_id}: negative control has non-empty gold")

        seed_id = raw.get("seed_case_id")
        if seed_id is not None and (isinstance(seed_id, bool) or not isinstance(seed_id, int)):
            raise ValueError(f"{row_id}: invalid seed_case_id")

        output.append(
            GoldRow(
                id=row_id,
                suite=suite,
                query=query,
                query_class_expected=query_class,
                answerable=answerable,
                gold_case_ids=case_ids,
                gold_chunk_ids=chunk_ids,
                gold_char_spans=spans,
                seed_case_id=seed_id,
                tags=raw.get("tags", []),
                verified=verified,
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


def normalise_span(value: Any) -> dict | None:
    if isinstance(value, dict):
        doc = value.get("doc", value.get("file_path"))
        start = value.get("start", value.get("char_start"))
        end = value.get("end", value.get("char_end"))
        if start is None or end is None:
            span = value.get("span")
            if isinstance(span, (list, tuple)) and len(span) == 2:
                start, end = span
        if doc is None and start is None and end is None:
            return None
    elif isinstance(value, (list, tuple)) and len(value) == 3:
        doc, start, end = value
    else:
        raise ValueError(f"Invalid retrieved span: {value!r}")

    if (
        not isinstance(doc, str)
        or not doc
        or isinstance(start, bool)
        or isinstance(end, bool)
        or not isinstance(start, int)
        or not isinstance(end, int)
        or start < 0
        or end <= start
    ):
        raise ValueError(f"Invalid retrieved span: {value!r}")

    return {"doc": doc, "start": start, "end": end}


def normalise_prediction(
    gold: GoldRow,
    result: Any,
    elapsed_ms: float,
) -> Prediction:
    if isinstance(result, list):
        result = {"hits": result}
    if not isinstance(result, dict):
        raise TypeError("Adapter must return a dict or a list of hits")
    hits = result.get("hits") or []
    if not isinstance(hits, list):
        raise TypeError("Adapter hits must be a list")

    case_ids = list(result.get("retrieved_case_ids") or [])
    chunk_ids = list(result.get("retrieved_chunk_ids") or [])
    spans = list(result.get("retrieved_spans") or [])

    for hit in hits:
        if not isinstance(hit, dict):
            raise TypeError("Every hit must be a dictionary")
        case_id = hit.get("case_id")
        chunk_id = hit.get("chunk_id")
        span = normalise_span(hit)
        if case_id is not None:
            case_ids.append(case_id)
        if chunk_id is not None:
            chunk_ids.append(chunk_id)
        if span is not None:
            spans.append(span)

    for value in case_ids:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"Invalid retrieved case ID: {value!r}")
    for value in chunk_ids:
        if not isinstance(value, str) or not value:
            raise ValueError(f"Invalid retrieved chunk ID: {value!r}")

    clean_spans = [normalise_span(v) for v in spans]
    clean_spans = [v for v in clean_spans if v is not None]

    arms = result.get("arms_used") or []
    diagnostics = result.get("arm_diagnostics") or []
    if not isinstance(arms, list) or not all(isinstance(item, str) for item in arms):
        raise ValueError("arms_used must be a list of strings")
    if not isinstance(diagnostics, list):
        raise ValueError("arm_diagnostics must be a list")

    unique_cases = list(dict.fromkeys(case_ids))
    unique_chunks = list(dict.fromkeys(chunk_ids))
    spans_tuple: list[tuple[str, int, int]] = [
        (s["doc"], s["start"], s["end"]) for s in clean_spans
    ]
    empty = not (unique_cases or unique_chunks or spans_tuple)

    return Prediction(
        id=gold.id,
        retrieved_case_ids=unique_cases,
        retrieved_chunk_ids=unique_chunks,
        retrieved_spans=spans_tuple,
        query_class_actual=result.get("query_class_actual"),
        arms_used=arms,
        arm_diagnostics=diagnostics,
        latency_ms=elapsed_ms,
        empty=empty,
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


def write_predictions(
    path: Path,
    predictions: list[Prediction],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for prediction in predictions:
            row = asdict(prediction)
            row["retrieved_spans"] = [
                {"doc": doc, "start": start, "end": end}
                for doc, start, end in prediction.retrieved_spans
            ]
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def score_slice(
    gold_rows: list[GoldRow],
    predictions: list[Prediction],
    top_k: int,
) -> dict:
    pairs = list(zip(gold_rows, predictions, strict=True))
    positive = [p for p in pairs if p[0].answerable]
    case_pairs = [p for p in positive if p[0].gold_case_ids]
    chunk_pairs = [p for p in positive if p[0].gold_chunk_ids]
    span_pairs = [p for p in positive if p[0].gold_char_spans is not None]

    case_recalls = [
        case_recall_at_k(g.gold_case_ids, p.retrieved_case_ids, top_k) for g, p in case_pairs
    ]
    case_ndcgs = [
        case_ndcg_at_k(g.gold_case_ids, p.retrieved_case_ids, top_k) for g, p in case_pairs
    ]
    case_mrrs = [mrr(g.gold_case_ids, p.retrieved_case_ids) for g, p in case_pairs]
    chunk_recalls = [
        chunk_recall_at_k(g.gold_chunk_ids, p.retrieved_chunk_ids, top_k) for g, p in chunk_pairs
    ]
    span_recalls = [
        char_span_recall(g.gold_char_spans or [], p.retrieved_spans[:top_k]) for g, p in span_pairs
    ]
    span_precisions = [
        char_span_precision(g.gold_char_spans or [], p.retrieved_spans[:top_k])
        for g, p in span_pairs
    ]

    rows = [
        {
            "retrieved_case_ids": p.retrieved_case_ids,
            "retrieved_chunk_ids": p.retrieved_chunk_ids,
            "retrieved_spans": p.retrieved_spans,
            "answerable": g.answerable,
        }
        for g, p in pairs
    ]

    return {
        "n": len(pairs),
        "n_answerable": len(positive),
        "n_negative": len(pairs) - len(positive),
        "n_case_scored": len(case_pairs),
        "n_chunk_scored": len(chunk_pairs),
        "n_span_scored": len(span_pairs),
        f"case_recall@{top_k}": mean(case_recalls),
        f"case_ndcg@{top_k}": mean(case_ndcgs),
        "case_mrr": mean(case_mrrs),
        f"chunk_recall@{top_k}": mean(chunk_recalls),
        f"char_span_recall@{top_k}": mean(span_recalls),
        f"char_span_precision@{top_k}": mean(span_precisions),
        "empty_result_rate": empty_result_rate(rows),
        "refusal_accuracy": refusal_accuracy(rows),
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
            f"Examples: {unverified[:10]}. "
            "Review gold or explicitly pass --allow-unverified."
        )

    corpus_hashes = {row.corpus_hash for row in gold_rows}
    if len(corpus_hashes) != 1:
        raise ValueError("Gold contains multiple corpus fingerprints")

    expected_hash = next(iter(corpus_hashes))
    gold_hash = hashlib.sha256(gold_path.read_bytes()).hexdigest()

    factory = import_factory(adapter_spec)
    runtime = cast(
        EvalRuntime,
        factory(
            suite=suite,
            config_name=config_name,
            exact_baseline=exact_baseline,
            no_graph=no_graph,
            rrf_k=rrf_k,
            overrides={},
        ),
    )

    predictions = []
    try:
        actual_hash = runtime.corpus_fingerprint()
        if actual_hash != expected_hash:
            raise RuntimeError(
                "CORPUS HASH MISMATCH. "
                f"Gold={expected_hash}; actual={actual_hash}. "
                "Refusing to score."
            )

        for index, gold in enumerate(gold_rows, start=1):
            started = time.perf_counter()
            result = runtime.retrieve(gold.query, top_k=top_k)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            prediction = normalise_prediction(gold, result, elapsed_ms)
            predictions.append(prediction)

            if index % 25 == 0 or index == len(gold_rows):
                print(
                    f"Evaluated {index}/{len(gold_rows)}",
                    file=sys.stderr,
                )
    finally:
        close = getattr(runtime, "close", None)
        if callable(close):
            close()

    grouped: dict[str, list[tuple[GoldRow, Prediction]]] = {}
    for gold, prediction in zip(gold_rows, predictions, strict=True):
        label = gold.query_class_expected if gold.answerable else "negative_control"
        grouped.setdefault(label, []).append((gold, prediction))

    per_class = {}
    for label, pairs in sorted(grouped.items()):
        per_class[label] = score_slice(
            [gold for gold, _ in pairs],
            [prediction for _, prediction in pairs],
            top_k,
        )

    positive_labels = [label for label in per_class if label != "negative_control"]

    def macro(key: str) -> float | None:
        values = [
            per_class[label].get(key)
            for label in positive_labels
            if per_class[label].get(key) is not None
        ]
        return mean(values)

    report = {
        "schema_version": 1,
        "suite": suite,
        "config": config_name,
        "top_k": top_k,
        "exact_baseline": exact_baseline,
        "no_graph": no_graph,
        "rrf_k": rrf_k,
        "n_queries": len(gold_rows),
        "n_verified": len(gold_rows) - len(unverified),
        "n_unverified": len(unverified),
        "corpus_hash": expected_hash,
        "gold_hash": gold_hash,
        "per_class": per_class,
        "macro_positive": {
            f"case_recall@{top_k}": macro(f"case_recall@{top_k}"),
            f"case_ndcg@{top_k}": macro(f"case_ndcg@{top_k}"),
            f"chunk_recall@{top_k}": macro(f"chunk_recall@{top_k}"),
            f"char_span_recall@{top_k}": macro(f"char_span_recall@{top_k}"),
            f"char_span_precision@{top_k}": macro(f"char_span_precision@{top_k}"),
        },
        "latency": latency_percentiles([p.latency_ms for p in predictions]),
    }

    atomic_write_json(out_path, report)
    prediction_path = out_path.with_name(out_path.stem + ".predictions.jsonl")
    write_predictions(prediction_path, predictions)

    print(json.dumps(report, indent=2))
    print(f"Metrics: {out_path}")
    print(f"Predictions: {prediction_path}")
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
    parser.add_argument("--no-graph", action="store_true")
    parser.add_argument("--allow-unverified", action="store_true")
    parser.add_argument("--rrf-k", type=int)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.top_k < 1:
            raise ValueError("--top-k must be positive")
        evaluate(
            gold_path=args.gold,
            out_path=args.out,
            adapter_spec=args.adapter,
            config_name=args.config,
            suite=args.suite,
            top_k=args.top_k,
            exact_baseline=args.exact_baseline,
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
