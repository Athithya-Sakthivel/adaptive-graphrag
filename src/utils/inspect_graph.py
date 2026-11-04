#!/usr/bin/env python3
"""inspect_graph.py — pre-load verification of graph construction.

Reads graph/citations_filtered.parquet, corpus/opinions.parquet, and
manifest/plan.jsonl WITHOUT touching the database or embedder.

Answers:
  1. How many edges exist in the source parquet?
  2. How many edges have BOTH endpoints in the current plan?
  3. Of those, how many will actually survive load_edges' filter?
  4. Degree distribution of the surviving edges
  5. Sample edges that will load / will not load (with reason)
  6. Sanity: are there any endpoints that aren't in the corpus?
  7. Sanity: self-loops, duplicates

Use this BEFORE `cli load` to know whether to expect edges=0 or edges=N.
At CORPUS_ROWS=5, edges=0 is expected. At CORPUS_ROWS=500, expect 100-300.

Usage:
    python3 src/utils/inspect_graph.py --data-dir /workspace/tmp/data
    python3 src/utils/inspect_graph.py --data-dir /workspace/tmp/data --json

Exit codes:
    0 — clean, edges should load as shown
    1 — warnings (e.g. zero edges will load, likely a sampling issue)
    2 — critical (dangling edges, self-loops, etc.)
    3 — setup error (missing files)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


class Reporter:
    def __init__(self, json_mode: bool = False) -> None:
        self.json_mode = json_mode
        self.critical: list[str] = []
        self.warnings: list[str] = []

    def section(self, name: str) -> None:
        if not self.json_mode:
            print(f"\n{'=' * 72}\n{name}\n{'=' * 72}")

    def check(self, name: str, value: Any, ok: bool | None = None) -> None:
        if not self.json_mode:
            mark = "PASS" if ok is True else "FAIL" if ok is False else "INFO"
            print(f"  [{mark:4s}] {name}: {value}")

    def finding(self, msg: str, critical: bool = False) -> None:
        if critical:
            self.critical.append(msg)
            if not self.json_mode:
                print(f"  [CRIT] {msg}")
        else:
            self.warnings.append(msg)
            if not self.json_mode:
                print(f"  ->   {msg}")


R = Reporter()


def load_plan_ids(data_dir: Path) -> set[int]:
    """Read the current plan.jsonl to know which case_ids are being loaded."""
    plan_path = data_dir / "manifest" / "plan.jsonl"
    if not plan_path.is_file():
        raise FileNotFoundError(f"missing {plan_path}")
    ids: set[int] = set()
    for line in plan_path.read_text().splitlines():
        if not line.strip():
            continue
        ids.add(int(json.loads(line)["case_id"]))
    return ids


def load_corpus_ids(data_dir: Path) -> set[int]:
    corpus = data_dir / "raw" / "corpus" / "opinions.parquet"
    if not corpus.is_file():
        raise FileNotFoundError(f"missing {corpus}")
    df = pd.read_parquet(corpus, columns=["cl_opinion_id"])
    return {int(x) for x in df["cl_opinion_id"].dropna().tolist()}


def load_edges(data_dir: Path) -> pd.DataFrame:
    edges = data_dir / "raw" / "graph" / "citations_filtered.parquet"
    if not edges.is_file():
        raise FileNotFoundError(f"missing {edges}")
    return pd.read_parquet(edges)


def inspect(data_dir: Path) -> dict:
    R.section("1. Source inventory")
    plan_ids = load_plan_ids(data_dir)
    corpus_ids = load_corpus_ids(data_dir)
    edges = load_edges(data_dir)

    R.check("plan case_ids (to load)", len(plan_ids))
    R.check("corpus case_ids (available)", len(corpus_ids))
    R.check("parquet edges (total)", len(edges))

    R.section("2. Endpoint integrity (against full corpus)")
    src_not_in_corpus = int((~edges["citing_opinion_id"].isin(corpus_ids)).sum())
    tgt_not_in_corpus = int((~edges["cited_opinion_id"].isin(corpus_ids)).sum())
    R.check("edges citing_id NOT in corpus", src_not_in_corpus, ok=(src_not_in_corpus == 0))
    R.check("edges cited_id  NOT in corpus", tgt_not_in_corpus, ok=(tgt_not_in_corpus == 0))
    if src_not_in_corpus or tgt_not_in_corpus:
        R.finding(
            f"{src_not_in_corpus + tgt_not_in_corpus} dangling endpoints — "
            f"the parquet is corrupt, re-run download_raw_data.py",
            critical=True,
        )

    R.section("3. Load-time filter (endpoints in plan)")
    src_in_plan = edges["citing_opinion_id"].isin(plan_ids)
    tgt_in_plan = edges["cited_opinion_id"].isin(plan_ids)
    both_in = int((src_in_plan & tgt_in_plan).sum())
    src_only = int((src_in_plan & ~tgt_in_plan).sum())
    tgt_only = int((~src_in_plan & tgt_in_plan).sum())
    neither = int((~src_in_plan & ~tgt_in_plan).sum())

    R.check(
        "edges with BOTH endpoints in plan (will load)",
        both_in,
        ok=(both_in > 0),
    )
    R.check("edges with only SOURCE in plan (dropped)", src_only)
    R.check("edges with only TARGET in plan (dropped)", tgt_only)
    R.check("edges with NEITHER endpoint in plan (dropped)", neither)

    if both_in == 0:
        R.finding(
            f"0 edges will load at {len(plan_ids)} cases. This is the "
            f"manifest plan being too small, not a code bug. "
            f"Expected edges-to-load at N cases ≈ {len(edges)} × "
            f"({len(plan_ids)}/{len(corpus_ids)})² ≈ "
            f"{int(len(edges) * (len(plan_ids) / len(corpus_ids)) ** 2)}. "
            f"Raise CORPUS_ROWS or use a stratified sample.",
            critical=False,
        )

    R.section("4. What will actually be in case_edges")
    if both_in > 0:
        loaded = edges[src_in_plan & tgt_in_plan]
        out_deg = loaded.groupby("citing_opinion_id").size()
        in_deg = loaded.groupby("cited_opinion_id").size()
        R.check("edge count", len(loaded))
        R.check("distinct citers (sources)", len(out_deg))
        R.check("distinct cited (targets)", len(in_deg))
        R.check(
            "out-degree p50/p95/max",
            [
                int(out_deg.median()),
                int(out_deg.quantile(0.95)),
                int(out_deg.max()),
            ],
        )
        R.check(
            "in-degree p50/p95/max",
            [
                int(in_deg.median()),
                int(in_deg.quantile(0.95)),
                int(in_deg.max()),
            ],
        )
        # Sample
        R.check("sample edges (first 5)", "")
        for _, row in loaded.head(5).iterrows():
            if not R.json_mode:
                print(
                    f"      {int(row['citing_opinion_id'])} → "
                    f"{int(row['cited_opinion_id'])}  "
                    f"(depth={row.get('depth')})"
                )

    R.section("5. Sanity checks")
    self_loops = int((edges["citing_opinion_id"] == edges["cited_opinion_id"]).sum())
    dupes = int(edges.duplicated(subset=["citing_opinion_id", "cited_opinion_id"]).sum())
    R.check("self-loops", self_loops, ok=(self_loops == 0))
    R.check("duplicate (citing, cited) pairs", dupes)
    if self_loops:
        R.finding(f"{self_loops} self-loops in parquet", critical=True)

    return {
        "plan_ids": len(plan_ids),
        "corpus_ids": len(corpus_ids),
        "parquet_edges": len(edges),
        "edges_will_load": both_in,
        "edges_dropped_src_not_in_plan": src_only,
        "edges_dropped_tgt_not_in_plan": tgt_only,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data-dir", type=Path, default=Path("/workspace/tmp/data"))
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    R.json_mode = bool(args.json)
    if not args.data_dir.is_dir():
        print(f"error: {args.data_dir} does not exist", file=sys.stderr)
        return 3

    try:
        stats = inspect(args.data_dir)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3

    if R.json_mode:
        print(
            json.dumps(
                {
                    "stats": stats,
                    "critical": R.critical,
                    "warnings": R.warnings,
                },
                indent=2,
            )
        )
    else:
        print(f"\n{'=' * 72}")
        if R.critical:
            print(f"CRITICAL: {len(R.critical)} finding(s)")
            for c in R.critical:
                print(f"  - {c}")
            print("\nFix the parquet before running load.")
        elif stats.get("edges_will_load", 0) == 0:
            print("WARNING: 0 edges will load — sample is too small")
            for w in R.warnings:
                print(f"  - {w}")
        else:
            print(f"OK: {stats['edges_will_load']} edges will load into case_edges")
        print("=" * 72)

    return 2 if R.critical else (1 if stats.get("edges_will_load", 0) == 0 else 0)


if __name__ == "__main__":
    sys.exit(main())
