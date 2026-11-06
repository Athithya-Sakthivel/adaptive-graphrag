"""Deterministic stratified custom gold-set generator."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import random
import sys
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def import_callable(spec: str):
    if ":" not in spec:
        raise ValueError("Use module:function syntax")
    module_name, function_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    function = getattr(module, function_name, None)
    if not callable(function):
        raise TypeError(f"Invalid callable: {spec}")
    return function


def case_ids(values: Any, name: str) -> list[int]:
    if values is None:
        return []
    if not isinstance(values, (list, tuple, set)):
        raise ValueError(f"{name} must be a collection")
    result = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"Invalid {name} value: {value!r}")
        result.append(value)
    return list(dict.fromkeys(result))


def validate_candidate(raw: dict) -> dict:
    for field in ("case_id", "case_name", "court", "year", "indegree"):
        if field not in raw:
            raise ValueError(f"Missing candidate field: {field}")

    if isinstance(raw["case_id"], bool) or not isinstance(raw["case_id"], int):
        raise ValueError("case_id must be an integer")
    if not isinstance(raw["year"], int):
        raise ValueError("year must be an integer")
    if (
        isinstance(raw["indegree"], bool)
        or not isinstance(raw["indegree"], int)
        or raw["indegree"] < 0
    ):
        raise ValueError("indegree must be a non-negative integer")
    if not isinstance(raw["case_name"], str) or not raw["case_name"].strip():
        raise ValueError("Invalid case name")
    if not isinstance(raw["court"], str) or not raw["court"].strip():
        raise ValueError("Invalid court")

    row = dict(raw)
    for field in (
        "citer_case_ids",
        "overruling_citer_ids",
        "application_case_ids",
    ):
        row[field] = case_ids(row.get(field), field)
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


def stratum(row: dict) -> tuple[str, str, str]:
    return (
        row["court"].lower().strip(),
        year_bucket(row["year"]),
        degree_bucket(row["indegree"]),
    )


def stable_shuffle(values: list, seed_material: str):
    digest = hashlib.sha256(seed_material.encode()).digest()
    seed = int.from_bytes(digest[:8], "big")
    random.Random(seed).shuffle(values)


def stratified_sample(
    candidates: Sequence[dict],
    sample_size: int,
    *,
    seed: int,
    label: str,
):
    if sample_size < 0:
        raise ValueError("Sample size cannot be negative")

    by_stratum = defaultdict(list)
    seen = set()
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

    allocation = {key: 0 for key in keys}
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

    selected_rows = []
    population_counts = {}
    sample_counts = {}
    for key in keys:
        population_counts["|".join(key)] = len(by_stratum[key])
        take = allocation[key]
        sample_counts["|".join(key)] = take
        selected_rows.extend(by_stratum[key][:take])
    return selected_rows, population_counts, sample_counts


def build_pools(candidates: Sequence[dict]):
    pools = {
        "HOLDING": [],
        "LOOKUP": [],
        "VALIDITY": [],
        "APPLICATION": [],
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
        if row["indegree"] >= 1 and row["citer_case_ids"]:
            pools["LOOKUP"].append(row)
        if row["indegree"] >= 1 and row["overruling_citer_ids"]:
            pools["VALIDITY"].append(row)
        if (
            row["indegree"] >= 1
            and row["application_case_ids"]
            and isinstance(row.get("doctrine"), str)
            and row["doctrine"].strip()
            and row.get("doctrine_leading") is True
        ):
            pools["APPLICATION"].append(row)
        if row["indegree"] == 0:
            pools["negative_control"].append(row)
    return pools


def holding_questions(row: dict, paraphraser, model: str | None):
    if paraphraser is None:
        return [f"What did the court hold in {row['case_name']}?"]
    questions = paraphraser(row["holding_text"], n=3, model=model)
    if not isinstance(questions, (list, tuple)):
        raise TypeError("Paraphraser must return a list")
    result = list(
        dict.fromkeys(" ".join(q.split()) for q in questions if isinstance(q, str) and q.strip())
    )
    if not result:
        raise ValueError(f"No valid questions for case {row['case_id']}")
    return result[:3]


def make_gold(
    label: str,
    row: dict,
    corpus_hash: str,
    paraphraser,
    model: str | None,
):
    case_id = row["case_id"]
    tags = [
        label.lower(),
        f"court:{row['court']}",
        f"year:{year_bucket(row['year'])}",
        f"indegree:{degree_bucket(row['indegree'])}",
    ]
    candidates = None

    if label == "HOLDING":
        candidates = holding_questions(row, paraphraser, model)
        query = candidates[0]
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
    elif label == "APPLICATION":
        query = f"How has {row['doctrine']} been applied since {row['year']}?"
        gold_cases = row["application_case_ids"]
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

    result = {
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
    if candidates is not None:
        result["query_candidates"] = candidates
    return result


def build_gold(
    *,
    source,
    tier: str,
    seed: int,
    minimum_corpus_cases: int,
    paraphraser,
    model: str | None,
):
    corpus_count = source.corpus_case_count()
    if isinstance(corpus_count, bool) or not isinstance(corpus_count, int):
        raise TypeError("Invalid corpus case count")
    if corpus_count < minimum_corpus_cases:
        raise RuntimeError(f"Corpus has {corpus_count} cases; minimum is {minimum_corpus_cases}")

    corpus_hash = source.corpus_fingerprint()
    if (
        not isinstance(corpus_hash, str)
        or len(corpus_hash) != 64
        or any(c not in "0123456789abcdefABCDEF" for c in corpus_hash)
    ):
        raise ValueError("Invalid corpus fingerprint")

    candidates = source.load_candidates()
    if not isinstance(candidates, list):
        raise TypeError("load_candidates() must return a list")

    pools = build_pools(candidates)

    if tier == "smoke":
        sizes = {
            "HOLDING": 2,
            "LOOKUP": 2,
            "VALIDITY": 2,
            "APPLICATION": 2,
            "negative_control": 2,
        }
    else:
        sizes = {
            "HOLDING": 50,
            "LOOKUP": 50,
            "VALIDITY": 50,
            "APPLICATION": 50,
            "negative_control": 20,
        }

    gold = []
    manifest_classes = {}

    for label, size in sizes.items():
        selected, population, sample = stratified_sample(
            pools[label],
            size,
            seed=seed,
            label=label,
        )
        manifest_classes[label] = {
            "eligible_population": len({row["case_id"] for row in pools[label]}),
            "sample_size": size,
            "population_by_stratum": population,
            "sample_by_stratum": sample,
        }
        for row in selected:
            gold.append(make_gold(label, row, corpus_hash, paraphraser, model))

    gold.sort(key=lambda row: (row["query_class_expected"], row["id"]))
    ids = [row["id"] for row in gold]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate gold IDs")

    expected = sum(sizes.values())
    if len(gold) != expected:
        raise RuntimeError(f"Expected {expected} rows; created {len(gold)}")

    selection_sql = source.selection_sql()
    if not isinstance(selection_sql, str):
        raise TypeError("selection_sql() must return SQL text")

    manifest = {
        "schema_version": 1,
        "suite": "custom",
        "tier": tier,
        "seed": seed,
        "corpus_case_count": corpus_count,
        "corpus_hash": corpus_hash,
        "total_queries": len(gold),
        "verified_at_creation": False,
        "verification_policy": ("Each entry must be reviewed before verified=true."),
        "sampling_method": (
            "Proportional deterministic stratification by court, "
            "five-year year bucket, and citation in-degree."
        ),
        "llm_model": model if paraphraser else None,
        "llm_used": paraphraser is not None,
        "selection_sql": selection_sql,
        "classes": manifest_classes,
        "class_counts": dict(Counter(row["query_class_expected"] for row in gold)),
    }
    return gold, manifest


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def build_parser():
    parser = argparse.ArgumentParser(description="Build custom legal retrieval gold data.")
    parser.add_argument("--source", required=True)
    parser.add_argument(
        "--tier",
        choices=("smoke", "full"),
        default="smoke",
    )
    parser.add_argument("--seed", type=int, default=20261106)
    parser.add_argument("--min-cases", type=int, default=5000)
    parser.add_argument("--paraphraser")
    parser.add_argument("--model")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("src/evals/gold/custom_smoke.jsonl"),
    )
    parser.add_argument("--manifest", type=Path)
    return parser


def main():
    args = build_parser().parse_args()
    source = None
    try:
        source_factory = import_callable(args.source)
        source = source_factory()
        paraphraser = import_callable(args.paraphraser) if args.paraphraser else None
        if paraphraser and not args.model:
            raise ValueError("--model is required when --paraphraser is supplied")

        gold, manifest = build_gold(
            source=source,
            tier=args.tier,
            seed=args.seed,
            minimum_corpus_cases=args.min_cases,
            paraphraser=paraphraser,
            model=args.model,
        )
        write_jsonl(args.out, gold)

        manifest_path = args.manifest or args.out.with_name(args.out.stem + ".manifest.json")
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )

        print(json.dumps(manifest, indent=2))
        print(f"Gold: {args.out}")
        print(f"Manifest: {manifest_path}")
        print("REVIEW REQUIRED: verified=false on all generated rows.")
        return 0
    except Exception as exc:
        print(f"GOLD BUILD FAILED: {exc}", file=sys.stderr)
        return 1
    finally:
        if source is not None:
            close = getattr(source, "close", None)
            if callable(close):
                close()


if __name__ == "__main__":
    raise SystemExit(main())
