#!/usr/bin/env python3
"""
adaptive-graphrag: download_datasets.py

Builds the finalized US CourtListener GraphRAG dataset repository.

Verified facts encoded in this file (CourtListener schema, 2026-06-30 bulk):
  * Join hierarchy:
        opinions.cluster_id -> clusters.id
        clusters.docket_id  -> dockets.id
        dockets.court_id    -> courts.id
    `court_id` is NOT on opinion-clusters. Court metadata comes from the
    corpus (docketx.us-caselaw.court / .jurisdiction).

  * citation-map column names have varied across snapshots
    (`citing_opinion` vs `citing_opinion_id`); detected at runtime.

  * CourtListener bulk CSVs contain rows where multi-line fields
    (judges, case_name, attorneys) shift downstream columns. `csv.reader`
    cannot rejoin split records, so field-count validation produces
    false positives. The reliable invariant is: `id` is field 0, and a
    valid row has a positive integer there.

  * pd.read_csv(usecols=...) returns columns in CSV order, not usecols
    order. Arrow casts require identical column order — reorder explicitly
    before casting.

  * Intermediate Parquet (77M-row citation map, ~10M-row clean clusters)
    lives in `intermediate/` and is excluded from MANIFEST + HF upload.

Usage:
    python3 download_datasets.py
    python3 download_datasets.py --dir /workspace/tmp/data/raw --scale min
    python3 download_datasets.py --scale medium
    python3 download_datasets.py --scale max --force
    python3 download_datasets.py --skip-clusters
    python3 download_datasets.py --clean-stale --clean-raw

Requirements:
    pip install datasets huggingface_hub pandas pyarrow requests
    apt-get install lbzip2       # optional; 6x faster decompression
"""

from __future__ import annotations

import argparse
import bz2
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import importlib.util


def _require(pkgs: list[str]) -> None:
    missing = [p for p in pkgs if importlib.util.find_spec(p) is None]
    if missing:
        sys.stderr.write(
            "Missing Python packages: " + ", ".join(missing) + "\n"
            "Run: pip install datasets huggingface_hub pandas pyarrow requests\n"
        )
        sys.exit(1)


_require(["datasets", "huggingface_hub", "pandas", "pyarrow", "requests"])

import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import requests  # noqa: E402
from datasets import load_dataset  # noqa: E402


# ---------------------------------------------------------------------------
# stdlib csv field-size limit
# ---------------------------------------------------------------------------
# opinion-clusters has fields larger than the 128 KB default (case_name,
# judges, nature_of_suit). sys.maxsize overflows the C long on some builds;
# step down until csv accepts the value.
def _raise_csv_field_limit() -> int:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return limit
        except OverflowError:
            limit //= 10


_CSV_FIELD_LIMIT = _raise_csv_field_limit()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCALE_PRESETS = {"min": 5_000, "medium": 100_000, "max": 400_000}

DOCKETX_REPO = "docketx/us-caselaw"
HOJICHA_REPO = "feyninc/hojicha"

COURTLISTENER_BULK = "https://storage.courtlistener.com/bulk-data"
DEFAULT_BULK_SNAPSHOT = "2026-06-30"

CORPUS_BATCH = 5_000
PARQUET_READ_BATCH = 500_000
CLUSTERS_BATCH = 10_000

SCALE_PLACEHOLDER = "__SCALE_PLACEHOLDER__"
INTERMEDIATE_DIR = "intermediate"

STALE_DIRS = [
    "graph_overlay",
    "structured_metadata",
    "production_corpus",
    "eval_isolated/adalat-ai",
]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def log(msg: str) -> None:
    print(msg, flush=True)


def log_err(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _parquet_rows(path: Path) -> int:
    return pq.ParquetFile(path).metadata.num_rows


def _safe_int(v: Any) -> int | None:
    if v is None or v == "":
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def download_bz2(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        log(f"[skip] raw already present: {dest.name} "
            f"({dest.stat().st_size / 1e6:.0f} MB)")
        return dest

    log(f"[fetch] {url}")
    with requests.get(url, stream=True, timeout=900) as r:
        if r.status_code != 200:
            raise RuntimeError(
                f"HTTP {r.status_code} for {url}\n"
                f"Check --bulk-snapshot; quarterly releases at {COURTLISTENER_BULK}/"
            )
        total = int(r.headers.get("content-length", 0))
        done = last_log = 0
        with dest.open("wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
                done += len(chunk)
                if done - last_log >= 100 << 20:
                    last_log = done
                    log(f"  {done / 1e6:.0f} / {total / 1e6:.0f} MB"
                        if total else f"  {done / 1e6:.0f} MB")
    log(f"[ok] {dest.name}: {dest.stat().st_size / 1e6:.0f} MB")
    return dest


def decompress_bz2(raw_bz2: Path, out_csv: Path, min_bytes: int = 100 << 20) -> Path:
    """Decompress bz2 once. Prefers lbzip2 (parallel) over the stdlib."""
    if out_csv.exists() and out_csv.stat().st_size >= min_bytes:
        log(f"[skip] decompressed already present: {out_csv.name} "
            f"({out_csv.stat().st_size / 1e6:.0f} MB)")
        return out_csv

    if out_csv.exists():
        out_csv.unlink()

    lbzip2 = shutil.which("lbzip2")
    if lbzip2:
        log(f"[decompress] lbzip2 (parallel) -> {out_csv.name}")
        result = subprocess.run(
            [lbzip2, "-d", "-k", "-n", str(os.cpu_count() or 4), str(raw_bz2)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"lbzip2 failed: {result.stderr}")
        produced = raw_bz2.with_suffix("")
        if produced != out_csv:
            produced.replace(out_csv)
    else:
        log(f"[decompress] bz2 (single-threaded; install lbzip2 for 6x speed)")
        with bz2.open(str(raw_bz2), "rb") as src, out_csv.open("wb") as dst:
            shutil.copyfileobj(src, dst)

    log(f"[ok] decompressed: {out_csv.name} "
        f"({out_csv.stat().st_size / 1e6:.0f} MB)")
    return out_csv


def read_csv_header(csv_path: Path) -> list[str]:
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        return next(csv.reader(fh))


def iter_parquet_batches(path: Path, batch_size: int = PARQUET_READ_BATCH):
    """Yield pandas DataFrames from a Parquet file. Version-safe."""
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=batch_size):
        yield batch.to_pandas()


def stream_dicts_to_parquet(
    row_iter: Iterable[dict],
    out_file: Path,
    schema: pa.Schema,
    batch_size: int,
) -> int:
    writer: pq.ParquetWriter | None = None
    batch: list[dict] = []
    total = 0

    def flush() -> None:
        nonlocal batch, writer, total
        if not batch:
            return
        table = pa.Table.from_pylist(batch, schema=schema)
        if writer is None:
            writer = pq.ParquetWriter(out_file, schema)
        writer.write_table(table)
        total += len(batch)
        batch = []

    try:
        for row in row_iter:
            batch.append(row)
            if len(batch) >= batch_size:
                flush()
        flush()
    finally:
        if writer is not None:
            writer.close()

    if writer is None:
        pq.write_table(pa.Table.from_pylist([], schema=schema), out_file)

    return total


# ---------------------------------------------------------------------------
# 1. Corpus: docketx/us-caselaw (streamed)
# ---------------------------------------------------------------------------
CORPUS_SCHEMA = pa.schema([
    ("id", pa.string()),
    ("cl_opinion_id", pa.int64()),
    ("cluster_id", pa.int64()),
    ("opinion_type", pa.string()),
    ("text_kind", pa.string()),
    ("title", pa.string()),
    ("text", pa.string()),
    ("citation", pa.string()),
    ("court", pa.string()),
    ("date", pa.string()),
    ("jurisdiction", pa.string()),
    ("doc_type", pa.string()),
    ("license", pa.string()),
    ("source", pa.string()),
    ("retrieved_at", pa.string()),
])


def fetch_corpus(out_dir: Path, cap: int, force: bool) -> None:
    out_file = out_dir / "opinions.parquet"
    if out_file.exists() and not force:
        log(f"[skip] corpus already exists ({_parquet_rows(out_file):,} rows)")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"[fetch] {DOCKETX_REPO} (streaming cap={cap:,}, batch={CORPUS_BATCH:,})")

    ds = load_dataset(DOCKETX_REPO, split="train", streaming=True)

    stats = {"no_text": 0, "no_clid": 0, "dup": 0}
    seen: set[int] = set()

    def rows() -> Iterable[dict]:
        count = 0
        for row in ds:
            if count >= cap:
                break
            text = row.get("text")
            if not text or not isinstance(text, str) or not text.strip():
                stats["no_text"] += 1
                continue
            extra = row.get("extra") or {}
            cl_id = extra.get("cl_opinion_id") if isinstance(extra, dict) else None
            if not cl_id:
                stats["no_clid"] += 1
                continue
            cl_id = int(cl_id)
            if cl_id in seen:
                stats["dup"] += 1
                continue
            seen.add(cl_id)
            cluster_id = extra.get("cluster_id")
            count += 1
            yield {
                "id": row.get("id"),
                "cl_opinion_id": cl_id,
                "cluster_id": int(cluster_id) if cluster_id else None,
                "opinion_type": extra.get("opinion_type"),
                "text_kind": extra.get("text_kind"),
                "title": row.get("title"),
                "text": text,
                "citation": row.get("citation"),
                "court": row.get("court"),
                "date": row.get("date"),
                "jurisdiction": row.get("jurisdiction"),
                "doc_type": row.get("doc_type"),
                "license": row.get("license"),
                "source": row.get("source"),
                "retrieved_at": row.get("retrieved_at"),
            }
            if count % 10_000 == 0:
                log(f"  streamed {count:,} usable rows")

    n = stream_dicts_to_parquet(rows(), out_file, CORPUS_SCHEMA, CORPUS_BATCH)
    log(f"[ok] corpus: {n:,} rows -> {out_file}")
    if any(stats.values()):
        log(f"  skipped: {stats['no_text']:,} no-text, "
            f"{stats['no_clid']:,} no-cl_opinion_id, "
            f"{stats['dup']:,} duplicate")


# ---------------------------------------------------------------------------
# 2. Citation map: CSV -> intermediate Parquet -> filtered
# ---------------------------------------------------------------------------
CITATION_ALIASES = [
    ("citing_opinion_id", "cited_opinion_id"),
    ("citing_opinion", "cited_opinion"),
]

CITATION_SCHEMA = pa.schema([
    ("citing_opinion_id", pa.int64()),
    ("cited_opinion_id", pa.int64()),
    ("depth", pa.int32()),
])


def detect_citation_columns(header: list[str]) -> tuple[str, str]:
    for a, b in CITATION_ALIASES:
        if a in header and b in header:
            return a, b
    raise RuntimeError(
        f"citation-map CSV has no recognised citation columns.\n"
        f"header = {header}\n"
        f"expected one of: {CITATION_ALIASES}"
    )


def citations_csv_to_parquet(raw_bz2: Path, out_parquet: Path, force: bool) -> int:
    if out_parquet.exists() and not force:
        log(f"[skip] citation parquet already exists "
            f"({_parquet_rows(out_parquet):,} rows)")
        return _parquet_rows(out_parquet)

    log(f"[convert] {raw_bz2.name} -> {out_parquet.name}")
    csv_path = decompress_bz2(raw_bz2, raw_bz2.with_suffix(""))
    header = read_csv_header(csv_path)
    citing_col, cited_col = detect_citation_columns(header)
    log(f"  detected columns: {citing_col} -> {cited_col}")

    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    total = 0

    try:
        for chunk in pd.read_csv(
            csv_path,
            usecols=[citing_col, cited_col, "depth"],
            dtype={citing_col: "int64", cited_col: "int64", "depth": "int32"},
            chunksize=PARQUET_READ_BATCH,
            low_memory=False,
        ):
            if citing_col != "citing_opinion_id":
                chunk = chunk.rename(columns={
                    citing_col: "citing_opinion_id",
                    cited_col: "cited_opinion_id",
                })
            # pd.read_csv returns columns in CSV order, not usecols order.
            # Reorder explicitly so table.cast(CITATION_SCHEMA) succeeds.
            chunk = chunk[["citing_opinion_id", "cited_opinion_id", "depth"]]
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            table = table.cast(CITATION_SCHEMA)
            if writer is None:
                writer = pq.ParquetWriter(out_parquet, CITATION_SCHEMA)
            writer.write_table(table)
            total += len(chunk)
            if total % 5_000_000 == 0:
                log(f"  {total:,} rows converted")
    finally:
        if writer is not None:
            writer.close()

    if writer is None:
        pq.write_table(
            pa.Table.from_pylist([], schema=CITATION_SCHEMA), out_parquet,
        )

    try:
        csv_path.unlink()
        log(f"[clean] removed intermediate {csv_path.name}")
    except OSError:
        pass

    log(f"[ok] citation parquet: {total:,} rows")
    return total


def filter_citations(
    corpus_file: Path,
    citations_parquet: Path,
    out_file: Path,
    force: bool,
) -> int:
    if out_file.exists() and not force:
        log(f"[skip] filtered citations already exist "
            f"({_parquet_rows(out_file):,} edges)")
        return _parquet_rows(out_file)

    log("[filter] citations -> edges within sampled corpus")
    corpus = pd.read_parquet(corpus_file, columns=["cl_opinion_id"])
    keep = set(corpus["cl_opinion_id"].dropna().astype("int64").tolist())
    log(f"  corpus opinions: {len(keep):,}")

    out_file.parent.mkdir(parents=True, exist_ok=True)
    if not keep:
        pq.write_table(
            pa.Table.from_pylist([], schema=CITATION_SCHEMA), out_file,
        )
        log("[warn] no corpus opinions — wrote empty citations")
        return 0

    writer: pq.ParquetWriter | None = None
    kept = total = 0
    try:
        for chunk in iter_parquet_batches(citations_parquet):
            total += len(chunk)
            mask = (
                chunk["citing_opinion_id"].isin(keep)
                & chunk["cited_opinion_id"].isin(keep)
            )
            filtered = chunk.loc[mask]
            if filtered.empty:
                continue
            filtered = filtered[["citing_opinion_id", "cited_opinion_id", "depth"]]
            table = pa.Table.from_pandas(filtered, preserve_index=False)
            table = table.cast(CITATION_SCHEMA)
            if writer is None:
                writer = pq.ParquetWriter(out_file, CITATION_SCHEMA)
            writer.write_table(table)
            kept += len(filtered)
            if total % 5_000_000 == 0:
                log(f"  {total:,} scanned, {kept:,} kept")
    finally:
        if writer is not None:
            writer.close()

    if writer is None:
        pq.write_table(
            pa.Table.from_pylist([], schema=CITATION_SCHEMA), out_file,
        )

    log(f"[ok] citations_filtered: {kept:,} / {total:,} edges")
    return kept


# ---------------------------------------------------------------------------
# 3. Opinion-clusters: dynamic schema + id-only validation
#
# `court_id` is NOT on opinion-clusters — it lives on dockets, two joins away.
# Optional columns vary by export snapshot, so the schema is built from the
# actual header. Row validation checks only `int(row[0])`:
#   - id is always the first physical field
#   - a multi-line field downstream cannot shift row[0]
#   - the previous `len(row) != n_fields` check produced ~72% false positives
#     because CourtListener exports contain legitimate multi-line fields that
#     csv.reader cannot rejoin
# ---------------------------------------------------------------------------
CLUSTER_FIELD_TYPES: dict[str, pa.DataType] = {
    "id": pa.int64(),
    "case_name": pa.string(),
    "case_name_short": pa.string(),
    "date_filed": pa.string(),
    "precedential_status": pa.string(),
    "citation_count": pa.int64(),
    "nature_of_suit": pa.string(),
    "judges": pa.string(),
    "docket_id": pa.int64(),
}
CLUSTER_INT_FIELDS = {"id", "citation_count", "docket_id"}


def _cluster_schema(present: list[str]) -> pa.Schema:
    return pa.schema([(c, CLUSTER_FIELD_TYPES[c]) for c in present])


def clean_clusters_csv_to_parquet(
    raw_bz2: Path, out_parquet: Path, force: bool,
) -> tuple[int, list[str]]:
    if out_parquet.exists() and not force:
        rows = _parquet_rows(out_parquet)
        cols = [f.name for f in pq.read_schema(out_parquet)]
        log(f"[skip] clean clusters parquet already exists ({rows:,} rows)")
        return rows, cols

    log(f"[clean] {raw_bz2.name} -> clean Parquet (validate id is int)")
    csv_path = decompress_bz2(raw_bz2, raw_bz2.with_suffix(""))
    header = read_csv_header(csv_path)

    if "id" not in header:
        raise RuntimeError(
            f"opinion-clusters CSV has no 'id' column.\n"
            f"header[:12] = {header[:12]}"
        )

    present = [c for c in CLUSTER_FIELD_TYPES if c in header]
    missing = sorted(set(CLUSTER_FIELD_TYPES) - set(present))
    log(f"  columns present: {present}")
    if missing:
        log(f"  columns absent (skipped): {missing}")

    schema = _cluster_schema(present)
    idx = {name: header.index(name) for name in present}
    id_idx = idx["id"]

    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    batch: list[dict] = []
    total = kept = skipped = 0

    def flush() -> None:
        nonlocal batch, writer, kept
        if not batch:
            return
        table = pa.Table.from_pylist(batch, schema=schema)
        if writer is None:
            writer = pq.ParquetWriter(out_parquet, schema)
        writer.write_table(table)
        kept += len(batch)
        batch = []

    try:
        with csv_path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh)
            next(reader)  # header
            for row in reader:
                total += 1
                if total % 2_000_000 == 0:
                    log(f"  {total:,} scanned, {kept:,} kept, "
                        f"{skipped:,} skipped")

                # id is always field 0. Downstream multi-line fields cannot
                # shift it. If int(row[0]) fails, the row is genuinely
                # corrupt (unclosed quote swallowed the id).
                try:
                    id_val = int(row[0])
                except (ValueError, IndexError):
                    skipped += 1
                    continue
                if id_val <= 0:
                    skipped += 1
                    continue

                rec: dict[str, Any] = {"id": id_val}
                for name in present:
                    if name == "id":
                        continue
                    try:
                        v = row[idx[name]]
                    except IndexError:
                        v = None
                    rec[name] = _safe_int(v) if name in CLUSTER_INT_FIELDS else (v or None)

                batch.append(rec)
                if len(batch) >= CLUSTERS_BATCH:
                    flush()
            flush()
    finally:
        if writer is not None:
            writer.close()

    if writer is None:
        pq.write_table(pa.Table.from_pylist([], schema=schema), out_parquet)

    try:
        csv_path.unlink()
        log(f"[clean] removed intermediate {csv_path.name}")
    except OSError:
        pass

    log(f"[ok] clean clusters: {kept:,} kept, {skipped:,} skipped")
    return kept, present


def filter_clusters(
    corpus_file: Path,
    clean_clusters_parquet: Path,
    out_file: Path,
    force: bool,
) -> int:
    if out_file.exists() and not force:
        log(f"[skip] filtered clusters already exist "
            f"({_parquet_rows(out_file):,} clusters)")
        return _parquet_rows(out_file)

    log("[filter] clusters -> only clusters referenced by corpus")
    schema = pq.read_schema(clean_clusters_parquet)
    cols = [f.name for f in schema]

    corpus = pd.read_parquet(corpus_file, columns=["cluster_id"])
    keep = set(corpus["cluster_id"].dropna().astype("int64").tolist())
    log(f"  corpus clusters: {len(keep):,}")

    out_file.parent.mkdir(parents=True, exist_ok=True)
    if not keep:
        pq.write_table(pa.Table.from_pylist([], schema=schema), out_file)
        log("[warn] no corpus clusters — wrote empty clusters")
        return 0

    writer: pq.ParquetWriter | None = None
    kept = total = 0
    try:
        for chunk in iter_parquet_batches(clean_clusters_parquet):
            total += len(chunk)
            filtered = chunk.loc[chunk["id"].isin(keep)]
            if filtered.empty:
                continue
            filtered = filtered[cols]
            table = pa.Table.from_pandas(filtered, preserve_index=False)
            table = table.cast(schema)
            if writer is None:
                writer = pq.ParquetWriter(out_file, schema)
            writer.write_table(table)
            kept += len(filtered)
    finally:
        if writer is not None:
            writer.close()

    if writer is None:
        pq.write_table(pa.Table.from_pylist([], schema=schema), out_file)

    log(f"[ok] clusters_filtered: {kept:,} / {total:,}")
    return kept


# ---------------------------------------------------------------------------
# 4. Chunking ablation: hojicha
# ---------------------------------------------------------------------------
def fetch_hojicha(out_dir: Path, force: bool) -> None:
    out_file = out_dir / "corpus.parquet"
    if out_file.exists() and not force:
        log("[skip] hojicha already exists")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"[fetch] {HOJICHA_REPO} (corpus + questions)")
    for cfg in ("corpus", "questions"):
        ds = load_dataset(HOJICHA_REPO, cfg, split="train")
        df = ds.to_pandas()
        pq.write_table(
            pa.Table.from_pandas(df, preserve_index=False),
            out_dir / f"{cfg}.parquet",
        )
        log(f"[ok] hojicha/{cfg}: {len(df):,} rows")


# ---------------------------------------------------------------------------
# 5. Synthesize citation-lookup eval
# ---------------------------------------------------------------------------
def synthesize_eval(
    edges_file: Path, out_dir: Path, max_queries: int = 200,
) -> int:
    log(f"[synth] building up to {max_queries} citation-lookup queries")
    edges = pd.read_parquet(edges_file)
    if edges.empty:
        log("[warn] no edges after filtering — skipping synthesized eval")
        return 0

    in_degree = edges["cited_opinion_id"].value_counts().head(max_queries)
    queries: list[dict[str, Any]] = []
    for cited_op, _count in in_degree.items():
        citing = edges.loc[
            edges["cited_opinion_id"] == int(cited_op), "citing_opinion_id"
        ].tolist()
        queries.append({
            "query": f"Find cases citing opinion {int(cited_op)}",
            "eval_type": "citation_lookup",
            "cited_opinion_id": int(cited_op),
            "ground_truth_citing_ids": [int(x) for x in citing],
            "ground_truth_count": len(citing),
            "derivation": "courtlistener_citation_map_set_membership",
        })

    df = pd.DataFrame(queries)
    out_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pandas(df, preserve_index=False),
        out_dir / "queries.parquet",
    )
    log(f"[ok] synthesized_graph_qa: {len(df):,} queries")
    return len(df)


# ---------------------------------------------------------------------------
# Repo metadata templates
# ---------------------------------------------------------------------------
README_TEMPLATE = """---
license: cc0-1.0
language:
- en
tags:
- legal
- us-case-law
- rag
- graph-rag
- citation-graph
- adaptive-retrieval
size_categories:
- 1K<n<10K
configs:
- config_name: corpus_opinions
  data_files:
  - split: train
    path: corpus/opinions.parquet
- config_name: graph_citations
  data_files:
  - split: train
    path: graph/citations_filtered.parquet
- config_name: metadata_clusters
  data_files:
  - split: train
    path: metadata/clusters_filtered.parquet
- config_name: eval_hojicha_corpus
  data_files:
  - split: train
    path: eval_isolated/hojicha/corpus.parquet
- config_name: eval_hojicha_questions
  data_files:
  - split: train
    path: eval_isolated/hojicha/questions.parquet
- config_name: eval_synthesized_graph_qa
  data_files:
  - split: train
    path: eval_isolated/synthesized_graph_qa/queries.parquet
---

# adaptive-graphrag-datasets

US CourtListener GraphRAG dataset snapshot (scale=__SCALE_PLACEHOLDER__).

See DATA_SOURCES.md for attribution and MANIFEST.json for SHA-256.

## Scale presets

| Scale | Corpus rows | Approx. vectors (@ ~5 chunks/doc) |
|---|---|---|
| min | 5,000 | ~25K |
| medium | 100,000 | ~500K |
| max | 400,000 | ~2M |

Graph and metadata are filtered to the sampled corpus, so they scale with it.

## Join keys

    corpus.opinions.cl_opinion_id  <->  graph.citations.citing_opinion_id
    corpus.opinions.cl_opinion_id  <->  graph.citations.cited_opinion_id
    corpus.opinions.cluster_id     <->  metadata.clusters.id

    metadata.clusters.docket_id    ->   dockets.id   (not shipped)
    dockets.court_id               ->   courts.id    (court metadata is on corpus)

## Quick start

    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id="<your-username>/adaptive-graphrag-datasets-max",
        repo_type="dataset",
        local_dir="data/raw",
    )
"""

LICENSE_TEMPLATE = """CC0 1.0 Universal (CC0 1.0) Public Domain Dedication

- docketx/us-caselaw corpus: public-domain-edict-of-government
- CourtListener bulk data: Public Domain Mark 1.0
- feyninc/hojicha: CC BY 4.0 (retained as-is in eval_isolated/hojicha/)

https://creativecommons.org/publicdomain/zero/1.0/
"""

DATA_SOURCES_TEMPLATE = """# Upstream sources and attribution

| Directory | Upstream | License | Changes applied |
|---|---|---|---|
| corpus/opinions.parquet | HF: docketx/us-caselaw | public-domain-edict-of-government | Streamed, capped at scale preset, deduplicated on cl_opinion_id |
| graph/citations_filtered.parquet | CourtListener citation-map-__SNAPSHOT__ | Public Domain Mark 1.0 | bz2 -> Parquet -> filtered to edges where both endpoints are in the corpus |
| metadata/clusters_filtered.parquet | CourtListener opinion-clusters-__SNAPSHOT__ | Public Domain Mark 1.0 | bz2 -> clean Parquet (rows with non-integer `id` dropped) -> filtered to corpus clusters |
| eval_isolated/hojicha/ | HF: feyninc/hojicha | CC BY 4.0 | Unmodified, exported to parquet |
| eval_isolated/synthesized_graph_qa/ | derived from graph/citations_filtered.parquet | CC0 1.0 (derived) | 200 citation-lookup queries; ground truth = citing set |

Retrieved: 2026-11-02. CourtListener bulk snapshot: __SNAPSHOT__.

## Schema notes

`court_id` is **not** on `opinion-clusters`. CourtListener's hierarchy is:

    opinions.cluster_id -> clusters.id
    clusters.docket_id  -> dockets.id
    dockets.court_id    -> courts.id

Court metadata is taken from the corpus (`court`, `jurisdiction`) rather than
from the clusters export, so no `dockets` download is required.
"""


def write_readme(root: Path, scale: str) -> None:
    (root / "README.md").write_text(
        README_TEMPLATE.replace(SCALE_PLACEHOLDER, scale), encoding="utf-8",
    )
    log("[ok] README.md")


def write_license(root: Path) -> None:
    (root / "LICENSE").write_text(LICENSE_TEMPLATE, encoding="utf-8")
    log("[ok] LICENSE")


def write_data_sources(root: Path, snapshot: str) -> None:
    (root / "DATA_SOURCES.md").write_text(
        DATA_SOURCES_TEMPLATE.replace("__SNAPSHOT__", snapshot), encoding="utf-8",
    )
    log("[ok] DATA_SOURCES.md")


def write_manifest(root: Path, scale: str, cap: int, summary: dict[str, Any]) -> None:
    log("[manifest] writing sha256 + row counts")
    manifest: dict[str, Any] = {
        "meta": {
            "scale": scale,
            "corpus_cap": cap,
            "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "bulk_snapshot": DEFAULT_BULK_SNAPSHOT,
            "graph_summary": summary,
            "source_versions": {
                "corpus": f"{DOCKETX_REPO}@streamed",
                "graph": f"courtlistener/citation-map-{DEFAULT_BULK_SNAPSHOT}",
                "metadata": f"courtlistener/opinion-clusters-{DEFAULT_BULK_SNAPSHOT}",
                "hojicha": f"{HOJICHA_REPO}@latest",
                "synthesized_eval": "derived from graph",
            },
        },
        "files": {},
    }

    intermediate = root / INTERMEDIATE_DIR
    for dirpath, _, filenames in os.walk(root):
        p_dir = Path(dirpath)
        if intermediate == p_dir or intermediate in p_dir.parents:
            continue
        for fn in sorted(filenames):
            if not (fn.endswith(".parquet") or fn.endswith(".json")):
                continue
            p = p_dir / fn
            entry: dict[str, Any] = {
                "sha256": sha256_file(p),
                "bytes": p.stat().st_size,
            }
            if fn.endswith(".parquet"):
                try:
                    entry["rows"] = _parquet_rows(p)
                except Exception:
                    pass
            manifest["files"][str(p.relative_to(root))] = entry

    (root / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8",
    )
    log(f"[ok] MANIFEST.json ({len(manifest['files'])} files)")


def clean_stale(root: Path) -> None:
    log("[clean-stale] removing Indian-stack leftovers")
    removed = 0
    for rel in STALE_DIRS:
        p = root / rel
        if p.exists():
            shutil.rmtree(p)
            log(f"  removed {rel}/")
            removed += 1
    p = root / "graph" / "citations.parquet"
    if p.exists():
        p.unlink()
        log("  removed graph/citations.parquet (unfiltered)")
        removed += 1
    if removed == 0:
        log("  nothing to clean")


def clean_raw(root: Path) -> None:
    log("[clean-raw] removing intermediates and compressed sources")
    inter = root / INTERMEDIATE_DIR
    if inter.exists():
        shutil.rmtree(inter)
        log(f"  removed {INTERMEDIATE_DIR}/")
    for pat in ("*.csv", "*.csv.bz2"):
        for sub in ("graph", "metadata"):
            for p in (root / sub).glob(pat):
                p.unlink()
                log(f"  removed {p.relative_to(root)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build the adaptive-graphrag US CourtListener dataset repository.",
    )
    ap.add_argument("--dir", default="/workspace/adaptive-graphrag-datasets")
    ap.add_argument("--scale", choices=["min", "medium", "max"], default="medium")
    ap.add_argument("--force", action="store_true",
                    help="re-download / re-filter existing outputs")
    ap.add_argument("--bulk-snapshot", default=DEFAULT_BULK_SNAPSHOT)
    ap.add_argument("--clean-stale", action="store_true",
                    help="delete stale Indian-stack dirs before building")
    ap.add_argument("--clean-raw", action="store_true",
                    help="delete intermediates and .csv/.bz2 after success")
    ap.add_argument("--skip-clusters", action="store_true",
                    help="skip the opinion-clusters stage entirely")
    args = ap.parse_args()

    root = Path(args.dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    inter = root / INTERMEDIATE_DIR
    inter.mkdir(parents=True, exist_ok=True)

    cap = SCALE_PRESETS[args.scale]

    log(f"Target: {root}")
    log(f"Scale: {args.scale}  (corpus cap = {cap:,})")
    log(f"Bulk snapshot: {args.bulk_snapshot}")
    log(f"CSV field limit: {_CSV_FIELD_LIMIT:,} bytes")
    log("")

    if args.clean_stale:
        clean_stale(root)
        log("")

    failures: list[tuple[str, str]] = []
    summary: dict[str, Any] = {}

    # 1. Corpus — fatal if it fails
    try:
        fetch_corpus(root / "corpus", cap, args.force)
    except Exception as e:
        log_err(f"[error] corpus: {e}")
        return 1

    # 2. Citation graph
    try:
        raw_cit = download_bz2(
            f"{COURTLISTENER_BULK}/citation-map-{args.bulk_snapshot}.csv.bz2",
            root / "graph" / f"citation-map-{args.bulk_snapshot}.csv.bz2",
        )
        cit_parquet = inter / "citation_map.parquet"
        citations_csv_to_parquet(raw_cit, cit_parquet, args.force)
        summary["edges"] = filter_citations(
            root / "corpus" / "opinions.parquet",
            cit_parquet,
            root / "graph" / "citations_filtered.parquet",
            args.force,
        )
    except Exception as e:
        log_err(f"[error] citations: {e}")
        failures.append(("citations", str(e)))

    # 3. Opinion clusters
    if args.skip_clusters:
        log("[skip] clusters — requested via --skip-clusters")
        summary["clusters"] = 0
    else:
        try:
            raw_clu = download_bz2(
                f"{COURTLISTENER_BULK}/opinion-clusters-{args.bulk_snapshot}.csv.bz2",
                root / "metadata" / f"opinion-clusters-{args.bulk_snapshot}.csv.bz2",
            )
            clean_parquet = inter / "clusters_clean.parquet"
            clean_clusters_csv_to_parquet(raw_clu, clean_parquet, args.force)
            summary["clusters"] = filter_clusters(
                root / "corpus" / "opinions.parquet",
                clean_parquet,
                root / "metadata" / "clusters_filtered.parquet",
                args.force,
            )
        except Exception as e:
            log_err(f"[error] clusters: {e}")
            failures.append(("clusters", str(e)))

    # 4. Chunking ablation
    try:
        fetch_hojicha(root / "eval_isolated" / "hojicha", args.force)
    except Exception as e:
        log_err(f"[error] hojicha: {e}")
        failures.append(("hojicha", str(e)))

    # 5. Synthesized eval
    edges_path = root / "graph" / "citations_filtered.parquet"
    if edges_path.exists():
        try:
            summary["eval_queries"] = synthesize_eval(
                edges_path, root / "eval_isolated" / "synthesized_graph_qa",
            )
        except Exception as e:
            log_err(f"[error] synthesize_eval: {e}")
            failures.append(("synthesize_eval", str(e)))

    # 6. Repo metadata
    write_readme(root, args.scale)
    write_license(root)
    write_data_sources(root, args.bulk_snapshot)
    write_manifest(root, args.scale, cap, summary)

    # 7. Optional cleanup (after manifest so intermediates are not counted)
    if args.clean_raw and not failures:
        clean_raw(root)

    log("")
    if failures:
        log_err("===== FAILED =====")
        for name, err in failures:
            log_err(f"  - {name}: {err}")
        return 1

    log(f"Done. Repository at: {root}")
    edges = summary.get("edges", 0)
    if edges < 5_000:
        log("")
        log(f"[warn] graph is sparse ({edges:,} edges).")
        log("       --scale min is a pipeline smoke test only.")
        log("       Use --scale medium (or max) for a demo-able graph.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
