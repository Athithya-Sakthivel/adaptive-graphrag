"""Deterministic stratified custom gold-set generator.

Builds a reproducible evaluation set from the live database.
Pools are stratified by court, year bucket, and citation in-degree
to ensure the eval set represents the full corpus distribution.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import random
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol, cast

from retriever.classify import extract_case_reference


class GoldSource(Protocol):
    """Contract every gold-set source must satisfy."""

    def corpus_case_count(self) -> int: ...
    def corpus_fingerprint(self) -> str: ...
    def load_candidates(self) -> list[dict[str, Any]]: ...
    def close(self) -> None: ...


def import_callable(spec: str) -> Callable[..., Any]:
    if ":" not in spec:
        raise ValueError("Use module:function syntax")
    module_name, function_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    function = getattr(module, function_name, None)
    if not callable(function):
        raise TypeError(f"Invalid callable: {spec}")
    return function


def validate_candidate(raw: dict[str, Any]) -> dict[str, Any]:
    required = ("case_id", "case_name", "court", "year", "indegree")
    for field in required:
        if field not in raw:
            raise ValueError(f"Missing candidate field: {field}")

    row = dict(raw)
    for field in ("citer_case_ids", "overruling_citer_ids"):
        values = row.get(field)
        if values is None:
            row[field] = []
        elif not isinstance(values, (list, tuple, set)):
            raise ValueError(f"{field} must be a collection")
        else:
            row[field] = [int(v) for v in values if isinstance(v, int)]
    return row


def degree_bucket(degree: int) -> str:
    if degree == 0:
        return "0"
    if degree == 1:
        return "1"
    if degree <= 4:
        return "2-4"
    return "5+"


def year_bucket(year: int) -> str:
    start = (year // 5) * 5
    return f"{start}-{start + 4}"


def stratum(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        row["court"].lower().strip(),
        year_bucket(row["year"]),
        degree_bucket(row["indegree"]),
    )


def stable_shuffle(values: list[dict[str, Any]], seed_material: str) -> None:
    digest = hashlib.sha256(seed_material.encode()).digest()
    seed = int.from_bytes(digest[:8], "big")
    random.Random(seed).shuffle(values)


def stratified_sample(
    candidates: Sequence[dict[str, Any]],
    sample_size: int,
    *,
    seed: int,
    label: str,
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, int]]:
    if sample_size < 0:
        raise ValueError("Sample size cannot be negative")

    by_stratum: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    seen: set[int] = set()
    for row in candidates:
        if row["case_id"] in seen:
            continue
        seen.add(row["case_id"])
        by_stratum[stratum(row)].append(row)

    total = sum(len(values) for values in by_stratum.values())
    if total < sample_size:
        raise ValueError(
            f"{label}: requested {sample_size}; only {total} eligible distinct seed cases"
        )

    keys = sorted(by_stratum)
    for key in keys:
        stable_shuffle(by_stratum[key], f"{seed}|{label}|{key}")

    allocation = dict.fromkeys(keys, 0)
    while sum(allocation.values()) < sample_size:
        available = [key for key in keys if allocation[key] < len(by_stratum[key])]
        if not available:
            raise RuntimeError("Unable to complete stratified allocation")
        selected = max(
            available,
            key=lambda key: (
                sample_size * len(by_stratum[key]) / total - allocation[key],
                tuple(str(x) for x in key),
            ),
        )
        allocation[selected] += 1

    selected_rows: list[dict[str, Any]] = []
    population_counts: dict[str, int] = {}
    sample_counts: dict[str, int] = {}
    for key in keys:
        population_counts["|".join(key)] = len(by_stratum[key])
        take = allocation[key]
        sample_counts["|".join(key)] = take
        selected_rows.extend(by_stratum[key][:take])
    return selected_rows, population_counts, sample_counts


def name_round_trips(case_name: str) -> bool:
    """True if the classifier extracts something usable from the query.

    Names with commas/colons are accepted — the retriever's trigram
    fallback is designed to handle them. Names that the classifier
    cannot parse at all are rejected.
    """
    extracted = extract_case_reference(f"Which cases cite {case_name}?")
    return bool(extracted and extracted.strip())


def build_pools(candidates: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    pools: dict[str, list[dict[str, Any]]] = {
        "HOLDING": [],
        "LOOKUP": [],
        "VALIDITY": [],
        "negative_control": [],
    }
    for raw in candidates:
        row = validate_candidate(raw)

        if (
            row.get("holding_chunk_id")
            and isinstance(row.get("holding_text"), str)
            and row["holding_text"].strip()
        ):
            pools["HOLDING"].append(row)

        resolvable = name_round_trips(row["case_name"])

        if resolvable and row["indegree"] >= 1 and row["citer_case_ids"]:
            pools["LOOKUP"].append(row)

        if resolvable and row["indegree"] >= 1 and row["overruling_citer_ids"]:
            pools["VALIDITY"].append(row)

        if row["indegree"] == 0:
            pools["negative_control"].append(row)

    return pools


def make_gold(label: str, row: dict[str, Any], corpus_hash: str) -> dict[str, Any]:
    case_id = row["case_id"]
    tags = [
        label.lower(),
        f"court:{row['court']}",
        f"year:{year_bucket(row['year'])}",
        f"indegree:{degree_bucket(row['indegree'])}",
    ]

    if label == "HOLDING":
        holding = " ".join(row["holding_text"].split())
        for name in (row["case_name"], row.get("case_name_short") or ""):
            if name:
                holding = holding.replace(name, "")
        query = holding[:400].strip()
        if len(query) < 60:
            query = f"What did the court hold in {row['case_name']}?"
        gold_cases = [case_id]
        gold_chunks = [row["holding_chunk_id"]]
    elif label == "LOOKUP":
        query = f"Which cases cite {row['case_name']}?"
        gold_cases = row["citer_case_ids"]
        gold_chunks = []
    elif label == "VALIDITY":
        query = f"Has {row['case_name']} been overruled?"
        gold_cases = row["overruling_citer_ids"]
        gold_chunks = []
    elif label == "negative_control":
        query = f"Which cases cite {row['case_name']}?"
        gold_cases = []
        gold_chunks = []
    else:
        raise ValueError(label)

    answerable = label != "negative_control"
    if answerable and not (gold_cases or gold_chunks):
        raise ValueError(f"{label} produced empty positive gold for {case_id}")

    return {
        "id": f"{label.lower()}-{case_id}",
        "suite": "custom",
        "query": query,
        "query_class_expected": label,
        "answerable": answerable,
        "gold_case_ids": list(dict.fromkeys(gold_cases)),
        "gold_chunk_ids": list(dict.fromkeys(gold_chunks)),
        "gold_char_spans": None,
        "seed_case_id": case_id,
        "tags": tags,
        "verified": False,
        "corpus_hash": corpus_hash,
        "verification_source": "project_generated_unreviewed",
    }


def build_gold(
    *,
    source: GoldSource,
    tier: str,
    seed: int,
    minimum_corpus_cases: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    corpus_count = source.corpus_case_count()
    if corpus_count < minimum_corpus_cases:
        raise RuntimeError(f"Corpus has {corpus_count} cases; minimum is {minimum_corpus_cases}")

    corpus_hash = source.corpus_fingerprint()
    candidates = source.load_candidates()
    pools = build_pools(candidates)

    sizes = (
        {"HOLDING": 2, "LOOKUP": 2, "VALIDITY": 2, "negative_control": 2}
        if tier == "smoke"
        else {"HOLDING": 50, "LOOKUP": 50, "VALIDITY": 30, "negative_control": 20}
    )

    gold: list[dict[str, Any]] = []
    manifest_classes: dict[str, Any] = {}

    for label, size in sizes.items():
        if not pools[label]:
            continue
        actual_size = min(size, len(pools[label]))
        selected, population, sample = stratified_sample(
            pools[label],
            actual_size,
            seed=seed,
            label=label,
        )
        manifest_classes[label] = {
            "eligible_population": len({row["case_id"] for row in pools[label]}),
            "sample_size": actual_size,
            "population_by_stratum": population,
            "sample_by_stratum": sample,
        }
        for row in selected:
            gold.append(make_gold(label, row, corpus_hash))

    gold.sort(key=lambda row: (row["query_class_expected"], row["id"]))

    manifest = {
        "schema_version": 1,
        "suite": "custom",
        "tier": tier,
        "seed": seed,
        "corpus_case_count": corpus_count,
        "corpus_hash": corpus_hash,
        "total_queries": len(gold),
        "verified_at_creation": False,
        "sampling_method": (
            "Proportional deterministic stratification by court, year bucket, "
            "and citation in-degree."
        ),
        "classes": manifest_classes,
        "class_counts": dict(Counter(row["query_class_expected"] for row in gold)),
    }
    return gold, manifest


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build custom legal retrieval gold data.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--tier", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--seed", type=int, default=20261106)
    parser.add_argument("--min-cases", type=int, default=500)
    parser.add_argument("--out", type=Path, default=Path("src/evals/gold/custom_smoke.jsonl"))
    parser.add_argument("--manifest", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    source: GoldSource | None = None
    try:
        source_factory = import_callable(args.source)
        source = cast(GoldSource, source_factory())

        gold, manifest = build_gold(
            source=source,
            tier=args.tier,
            seed=args.seed,
            minimum_corpus_cases=args.min_cases,
        )
        write_jsonl(args.out, gold)

        manifest_path = args.manifest or args.out.with_name(args.out.stem + ".manifest.json")
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

        print(f"  corpus:     {manifest['corpus_case_count']} cases")
        print(f"  fingerprint: {manifest['corpus_hash'][:16]}...")
        print(f"  queries:    {manifest['total_queries']}")
        for label, info in manifest["classes"].items():
            print(
                f"    {label:<18} {info['sample_size']:>3} / {info['eligible_population']} eligible"
            )
        print(f"  gold:       {args.out}")
        print(f"  manifest:   {manifest_path}")
        print("  note:       verified=false on all rows (review before claiming verified)")
        return 0
    except Exception as exc:
        print(f"GOLD BUILD FAILED: {exc}", file=sys.stderr)
        return 1
    finally:
        if source is not None:
            source.close()


if __name__ == "__main__":
    raise SystemExit(main())
