#!/usr/bin/env python3
"""
inspect.py — deep inspection of data/raw/ before writing the indexing pipeline.

Reads every Parquet and CSV file under data/raw/, validates schema against the
data contract in docs/rag/datasets.md, measures the distributions the chunker
depends on, checks referential integrity of the citation graph, and recommends
a `tiny` preset for fast iteration.

Usage:
    python src/utils/inspect.py /workspace/tmp/data/raw
    python src/utils/inspect.py /workspace/tmp/data/raw --json > inspect.json

No writes. No mutations. Read-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EXPECTED: dict[str, dict[str, str]] = {
    "corpus/opinions.parquet": {
        "cl_opinion_id": "int64",
        "cluster_id": "int64",
        "opinion_type": "object",
        "text_kind": "object",
        "title": "object",
        "text": "object",
        "citation": "object",
        "court": "object",
        "date": "object",
        "jurisdiction": "object",
        "doc_type": "object",
    },
    "graph/citations_filtered.parquet": {
        "citing_opinion_id": "int64",
        "cited_opinion_id": "int64",
        "depth": "int32",
    },
    "metadata/clusters_filtered.parquet": {
        "id": "int64",
        "case_name": "object",
        "case_name_short": "object",
        "date_filed": "object",
        "precedential_status": "object",
        "citation_count": "int64",
        "nature_of_suit": "object",
        "judges": "object",
        "docket_id": "int64",
    },
}

PARENT_TARGET_TOK = 1200
PARENT_MAX_TOK = 1600
PARENT_MIN_TOK = 600
CHILD_TARGET_TOK = 350
CHILD_MAX_TOK = 450
CHILD_MIN_TOK = 180

CHARS_PER_TOKEN = 4.5


def est_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, int(len(text) / CHARS_PER_TOKEN))


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------


class Reporter:
    def __init__(self, json_mode: bool = False) -> None:
        self.json_mode = json_mode
        self.sections: list[dict] = []
        self.current: dict = {"section": "", "checks": [], "findings": []}

    def section(self, name: str) -> None:
        self.current = {"section": name, "checks": [], "findings": []}
        if not self.json_mode:
            print(f"\n{'=' * 72}\n{name}\n{'=' * 72}")

    def check(self, name: str, value, ok: bool | None = None, note: str = "") -> None:
        entry = {"name": name, "value": value, "ok": ok, "note": note}
        self.current["checks"].append(entry)
        if not self.json_mode:
            mark = "PASS" if ok is True else "FAIL" if ok is False else "INFO"
            print(f"  [{mark}] {name}: {value}" + (f"  ({note})" if note else ""))

    def finding(self, msg: str) -> None:
        self.current["findings"].append(msg)
        if not self.json_mode:
            print(f"  -> {msg}")

    def end(self) -> None:
        self.sections.append(self.current)

    def dump(self) -> dict:
        return {"sections": self.sections}


R = Reporter()


# ---------------------------------------------------------------------------
# 1. File presence and integrity
# ---------------------------------------------------------------------------


def inspect_files(root: Path) -> None:
    R.section("1. File presence and integrity")

    manifest_path = root / "MANIFEST.json"
    manifest: dict = {}
    manifest_exists = manifest_path.exists()
    if manifest_exists:
        try:
            manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError as exc:
            R.check("MANIFEST.json", f"invalid JSON: {exc}", ok=False)
            manifest = {}
            manifest_exists = False

        if manifest_exists:
            files = manifest.get("files", manifest) if isinstance(manifest, dict) else manifest
            R.check("MANIFEST.json entries", len(files))
    else:
        R.check("MANIFEST.json", "MISSING", ok=False)

    required = [
        "corpus/opinions.parquet",
        "graph/citations_filtered.parquet",
        "metadata/clusters_filtered.parquet",
        "eval_isolated/hojicha/corpus.parquet",
        "eval_isolated/hojicha/questions.parquet",
        "eval_isolated/synthesized_graph_qa/queries.parquet",
    ]

    for rel in required:
        p = root / rel
        if not p.exists():
            R.check(rel, "MISSING", ok=False)
            continue
        size_mb = p.stat().st_size / (1024 * 1024)
        sha = hashlib.sha256(p.read_bytes()).hexdigest()

        expected_sha = None
        if manifest_exists and isinstance(manifest, dict):
            if "files" in manifest and isinstance(manifest["files"], dict):
                entry = manifest["files"].get(rel)
                if isinstance(entry, dict):
                    expected_sha = entry.get("sha256")
                elif isinstance(entry, str):
                    expected_sha = entry
            else:
                entry = manifest.get(rel)
                if isinstance(entry, dict):
                    expected_sha = entry.get("sha256")
                elif isinstance(entry, str):
                    expected_sha = entry

        ok: bool | None = None
        note = f"{size_mb:.1f} MB"
        if expected_sha:
            ok = sha == expected_sha
            note += ", sha256 " + ("match" if ok else "MISMATCH")

        R.check(rel, f"{sha[:12]}...", ok=ok, note=note)


# ---------------------------------------------------------------------------
# 2. Parquet schema validation
# ---------------------------------------------------------------------------


def inspect_schema(root: Path) -> None:
    R.section("2. Parquet schema validation (vs datasets.md)")

    for rel, expected_cols in EXPECTED.items():
        p = root / rel
        if not p.exists():
            continue
        schema = pq.read_schema(p)
        actual = {f.name: str(f.type) for f in schema}
        missing = [c for c in expected_cols if c not in actual]
        extra = [c for c in actual if c not in expected_cols]

        if missing:
            R.check(f"{rel} missing columns", missing, ok=False)
        else:
            R.check(f"{rel} columns", f"{len(actual)} (all expected present)", ok=True)

        if extra:
            R.check(f"{rel} extra columns", extra, note="not in contract, may be safe")


# ---------------------------------------------------------------------------
# 3. Row counts and null coverage
# ---------------------------------------------------------------------------


def inspect_corpus(root: Path) -> pd.DataFrame | None:
    R.section("3. Corpus — row count, nulls, text distribution")

    p = root / "corpus/opinions.parquet"
    if not p.exists():
        return None

    df = pd.read_parquet(
        p,
        columns=[
            "cl_opinion_id",
            "cluster_id",
            "opinion_type",
            "text_kind",
            "text",
            "title",
            "citation",
            "court",
            "date",
        ],
    )
    R.check("corpus rows", len(df))

    for c in [
        "cl_opinion_id",
        "text",
        "title",
        "citation",
        "court",
        "date",
        "cluster_id",
        "opinion_type",
        "text_kind",
    ]:
        if c not in df.columns:
            continue
        nulls = df[c].isna().sum()
        pct = 100 * nulls / len(df)
        ok: bool | None = None
        if c in ("text", "cl_opinion_id"):
            ok = nulls == 0
        elif c in ("title", "citation", "court", "date", "opinion_type"):
            ok = pct < 5
        elif c == "cluster_id":
            ok = pct < 2
        R.check(f"corpus.{c} nulls", f"{nulls} ({pct:.2f}%)", ok=ok)

    dup_ids = int(df["cl_opinion_id"].duplicated().sum())
    R.check("duplicate cl_opinion_id", dup_ids, ok=(dup_ids == 0), note="contract requires unique")

    if "text_kind" in df.columns:
        tk = df["text_kind"].value_counts(dropna=False).to_dict()
        R.check("text_kind distribution", tk)
        n_kinds = df.groupby("cl_opinion_id")["text_kind"].nunique()
        multi = int((n_kinds > 1).sum())
        R.check(
            "opinions with >1 text_kind",
            multi,
            note="manifest must dedupe by original>combined>generated",
        )

    if "opinion_type" in df.columns:
        ot = df["opinion_type"].value_counts(dropna=False).head(10).to_dict()
        R.check("opinion_type top-10", ot)

    if "court" in df.columns:
        courts = df["court"].value_counts()
        R.check("court cardinality", int(len(courts)))
        R.check("court top-5", courts.head(5).to_dict())
        R.check("courts with >=5 opinions", int((courts >= 5).sum()))

    if "date" in df.columns:
        d = pd.to_datetime(df["date"], errors="coerce")
        valid = int(d.notna().sum())
        R.check("date parseable", f"{valid}/{len(df)}", ok=(valid / len(df) > 0.95))
        if valid:
            R.check("date range", f"{d.min().date()} .. {d.max().date()}")

    text = df["text"].fillna("")
    tokens = text.map(est_tokens)
    R.check("text tokens p50", int(tokens.median()))
    R.check("text tokens p95", int(tokens.quantile(0.95)))
    R.check("text tokens p99", int(tokens.quantile(0.99)))
    R.check("text tokens max", int(tokens.max()))

    can_parent = int((tokens >= PARENT_MIN_TOK).sum())
    pct = 100 * can_parent / len(df)
    R.check(
        "opinions >= parent_min_tokens",
        f"{can_parent} ({pct:.1f}%)",
        ok=(pct > 90),
        note="below this, chunker emits one short parent",
    )

    est_parents = tokens.map(lambda t: max(1, round(t / PARENT_TARGET_TOK)))
    est_children = est_parents * 3
    R.check("est chunks (total)", int((est_parents + est_children).sum()))
    R.check("est chunks per opinion p50", int((est_parents + est_children).median()))
    R.check("est chunks per opinion p95", int((est_parents + est_children).quantile(0.95)))

    return df


# ---------------------------------------------------------------------------
# 4. Graph inspection
# ---------------------------------------------------------------------------


def inspect_graph(root: Path, df_corpus: pd.DataFrame | None) -> None:
    R.section("4. Citation graph — integrity and degree distribution")

    p = root / "graph/citations_filtered.parquet"
    if not p.exists():
        return

    g = pd.read_parquet(p)
    R.check("graph edges", len(g))

    if df_corpus is not None:
        corpus_ids = set(df_corpus["cl_opinion_id"].astype("int64").tolist())
        src_missing = int((~g["citing_opinion_id"].isin(corpus_ids)).sum())
        tgt_missing = int((~g["cited_opinion_id"].isin(corpus_ids)).sum())
        R.check("edges with citing_id not in corpus", src_missing, ok=(src_missing == 0))
        R.check("edges with cited_id not in corpus", tgt_missing, ok=(tgt_missing == 0))

    self_loops = int((g["citing_opinion_id"] == g["cited_opinion_id"]).sum())
    R.check("self-loops", self_loops, ok=(self_loops == 0))

    dup_edges = int(g.duplicated(subset=["citing_opinion_id", "cited_opinion_id"]).sum())
    R.check("duplicate (citing, cited) pairs", dup_edges)

    out_deg = g.groupby("citing_opinion_id").size()
    in_deg = g.groupby("cited_opinion_id").size()
    R.check(
        "out-degree p50/p95/max",
        [int(out_deg.median()), int(out_deg.quantile(0.95)), int(out_deg.max())],
    )
    R.check(
        "in-degree p50/p95/max",
        [int(in_deg.median()), int(in_deg.quantile(0.95)), int(in_deg.max())],
    )

    if "depth" in g.columns:
        R.check("depth distribution", g["depth"].value_counts().head(5).to_dict())

    seeds = random.sample(list(out_deg.index), min(500, len(out_deg)))
    out_adj: dict = g.groupby("citing_opinion_id")["cited_opinion_id"].apply(list).to_dict()
    hop1_counts: list[int] = []
    hop2_counts: list[int] = []
    for s in seeds:
        h1 = out_adj.get(s, [])
        hop1_counts.append(len(h1))
        h2: set = set()
        for n in h1:
            h2.update(out_adj.get(n, []))
        hop2_counts.append(len(h2))

    if hop2_counts:
        R.check(
            "sampled 2-hop reach p50/p95/max",
            [
                int(pd.Series(hop2_counts).median()),
                int(pd.Series(hop2_counts).quantile(0.95)),
                int(max(hop2_counts)),
            ],
        )
        R.check(
            "seeds with 0 two-hop reach",
            sum(1 for c in hop2_counts if c == 0),
            ok=None,
            note="at tiny scale, expect many",
        )


# ---------------------------------------------------------------------------
# 5. Metadata inspection
# ---------------------------------------------------------------------------


def inspect_metadata(root: Path, df_corpus: pd.DataFrame | None) -> None:
    R.section("5. Cluster metadata — coverage and cardinality")

    p = root / "metadata/clusters_filtered.parquet"
    if not p.exists():
        return

    m = pd.read_parquet(p)
    R.check("clusters rows", len(m))

    if "precedential_status" in m.columns:
        ps = m["precedential_status"].value_counts(dropna=False).head(10).to_dict()
        R.check("precedential_status distribution", ps)

    if "nature_of_suit" in m.columns:
        R.check("nature_of_suit cardinality", int(m["nature_of_suit"].nunique()))

    if "date_filed" in m.columns:
        d = pd.to_datetime(m["date_filed"], errors="coerce")
        valid = int(d.notna().sum())
        R.check("date_filed parseable", f"{valid}/{len(m)}")
        if valid:
            R.check("date_filed range", f"{d.min().date()} .. {d.max().date()}")

    if "case_name" in m.columns:
        html = int(m["case_name"].fillna("").str.contains(r"<[a-z]+>", regex=True).sum())
        R.check("case_name with HTML markup", html, note="cosmetic, does not affect joins")

    if df_corpus is not None and "cluster_id" in df_corpus.columns:
        with_cluster = int(df_corpus["cluster_id"].notna().sum())
        in_meta = int(df_corpus["cluster_id"].isin(m["id"]).sum())
        R.check("corpus opinions with cluster_id", f"{with_cluster}/{len(df_corpus)}")
        R.check(
            "corpus clusters present in metadata",
            f"{in_meta}/{len(df_corpus)}",
            ok=(in_meta / len(df_corpus) > 0.95),
        )


# ---------------------------------------------------------------------------
# 6. Eval isolation check
# ---------------------------------------------------------------------------


def inspect_eval(root: Path) -> None:
    R.section("6. Eval isolation (must not contaminate production)")

    for rel in [
        "eval_isolated/hojicha/corpus.parquet",
        "eval_isolated/hojicha/questions.parquet",
        "eval_isolated/synthesized_graph_qa/queries.parquet",
    ]:
        p = root / rel
        if not p.exists():
            R.check(rel, "MISSING", ok=False)
            continue
        schema = pq.read_schema(p)
        n = pq.ParquetFile(p).metadata.num_rows
        R.check(rel, f"{n} rows, {len(schema)} cols", ok=True)

    hq = root / "eval_isolated/hojicha/corpus.parquet"
    corpus = root / "corpus/opinions.parquet"
    if hq.exists() and corpus.exists():
        h = pd.read_parquet(hq)
        c = pd.read_parquet(corpus, columns=["title"])
        if "title" in h.columns and "title" in c.columns:
            overlap = int(h["title"].isin(c["title"]).sum())
            R.check(
                "hojicha titles overlapping corpus titles",
                overlap,
                ok=(overlap == 0),
                note="eval must not appear in production corpus",
            )


# ---------------------------------------------------------------------------
# 7. Tiny preset recommendation
# ---------------------------------------------------------------------------


def recommend_tiny(df_corpus: pd.DataFrame | None, df_graph: pd.DataFrame | None) -> None:
    R.section("7. Preset recommendation (tiny, min, medium)")

    if df_corpus is None:
        return

    R.check("corpus row count", len(df_corpus))
    if df_graph is not None:
        R.check("graph edge count", len(df_graph))

    presets = [1000, 2000, 5000, 10000, len(df_corpus)]
    table: list[dict] = []

    for n in presets:
        if n > len(df_corpus):
            continue
        sample = df_corpus.head(n)

        # ids is computed unconditionally: it depends only on the sample,
        # not on df_graph. This eliminates the basedpyright "possibly
        # unbound" diagnostic at the second `if df_graph is not None` block.
        ids: set[int] = set(sample["cl_opinion_id"].astype("int64").tolist())

        types = int(sample["opinion_type"].nunique()) if "opinion_type" in sample else 0
        courts = int(sample["court"].nunique()) if "court" in sample else 0
        tokens = sample["text"].fillna("").map(est_tokens)
        est_parents = tokens.map(lambda t: max(1, round(t / PARENT_TARGET_TOK)))
        est_children = est_parents * 3
        est_chunks = int((est_parents + est_children).sum())

        retained = 0
        hop2_pct = 0.0
        if df_graph is not None:
            retained = int(
                (
                    df_graph["citing_opinion_id"].isin(ids) & df_graph["cited_opinion_id"].isin(ids)
                ).sum()
            )
            if retained > 0:
                sub = df_graph[
                    df_graph["citing_opinion_id"].isin(ids) & df_graph["cited_opinion_id"].isin(ids)
                ]
                out_adj: dict = (
                    sub.groupby("citing_opinion_id")["cited_opinion_id"].apply(list).to_dict()
                )
                rng = random.Random(42)
                seed_ids = list(out_adj.keys())
                seeds = rng.sample(seed_ids, min(100, len(seed_ids)))
                h2 = 0
                for s in seeds:
                    reach: set = set()
                    for n1 in out_adj.get(s, []):
                        reach.update(out_adj.get(n1, []))
                    if reach:
                        h2 += 1
                hop2_pct = 100 * h2 / len(seeds) if seeds else 0.0

        table.append(
            {
                "n": n,
                "opinion_types": types,
                "courts": courts,
                "graph_edges": retained,
                "est_chunks": est_chunks,
                "seeds_with_2hop_pct": round(hop2_pct, 1),
            }
        )

    if R.json_mode:
        R.check("preset_table", table)
    else:
        print(
            f"\n  {'preset':>8} {'types':>6} {'courts':>7} {'edges':>7} {'chunks':>8} {'2hop%':>7}"
        )
        print(f"  {'-' * 8:>8} {'-' * 6:>6} {'-' * 7:>7} {'-' * 7:>7} {'-' * 8:>8} {'-' * 7:>7}")
        for row in table:
            print(
                f"  {row['n']:>8} {row['opinion_types']:>6} {row['courts']:>7} "
                f"{row['graph_edges']:>7} {row['est_chunks']:>8} "
                f"{row['seeds_with_2hop_pct']:>6}%"
            )

    tiny = next((r for r in table if r["n"] == 2000), None)
    min_ = next((r for r in table if r["n"] == 5000), None)

    if tiny is not None:
        if tiny["graph_edges"] > 500 and tiny["seeds_with_2hop_pct"] > 50:
            R.finding(
                f"tiny=2000 is VIABLE: {tiny['graph_edges']} edges, "
                f"{tiny['seeds_with_2hop_pct']}% seeds with 2-hop reach, "
                f"~{tiny['est_chunks']} chunks"
            )
        else:
            R.finding(
                f"tiny=2000 is TOO SPARSE: {tiny['graph_edges']} edges, "
                f"{tiny['seeds_with_2hop_pct']}% 2-hop. Use min=5000 instead."
            )

    if min_ is not None:
        R.finding(
            f"min=5000: {min_['graph_edges']} edges, "
            f"~{min_['est_chunks']} chunks, "
            f"{min_['seeds_with_2hop_pct']}% 2-hop reach"
        )

    if tiny is not None and tiny["graph_edges"] > 500 and tiny["seeds_with_2hop_pct"] > 50:
        R.finding(
            "Recommended iteration ladder: tiny=2000 (dev) → min=5000 (smoke) → medium=100K (demo)"
        )
    else:
        R.finding(
            "Recommended iteration ladder: "
            "min=5000 (dev + smoke) → medium=100K (demo). "
            "No tiny preset — graph too sparse below 5,000 rows."
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default="/workspace/tmp/data/raw")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    R.json_mode = bool(args.json)
    root = Path(args.root)
    if not root.exists():
        print(f"error: {root} does not exist", file=sys.stderr)
        sys.exit(1)
    if not R.json_mode:
        print(f"inspecting: {root}")

    inspect_files(root)
    R.end()

    inspect_schema(root)
    R.end()

    df_corpus = inspect_corpus(root)
    R.end()

    gp = root / "graph/citations_filtered.parquet"
    df_graph: pd.DataFrame | None = pd.read_parquet(gp) if gp.exists() else None

    inspect_graph(root, df_corpus)
    R.end()

    inspect_metadata(root, df_corpus)
    R.end()

    inspect_eval(root)
    R.end()

    recommend_tiny(df_corpus, df_graph)
    R.end()

    if R.json_mode:
        print(json.dumps(R.dump(), indent=2, default=str))
    else:
        print(f"\n{'=' * 72}\nDONE\n{'=' * 72}")


if __name__ == "__main__":
    main()
