# `/workspace/data` — Architecture and Data Contract

## 1. Purpose

This directory is a **two-tier dataset pipeline** for the `adaptive-graphrag` US CourtListener portfolio project. It separates two concerns that are usually conflated:

- **Producer** (`archive/`) — assembles the dataset from upstream sources, publishes it to Hugging Face, and runs on your machine.
- **Consumer** (`download_raw_data.py`, `run.sh`) — pulls a published snapshot into `data/raw/` for downstream ingestion (chunking, indexing, eval).

The split means a reviewer clones this repo, runs **one script**, and has a byte-identical dataset — no CourtListener CSV wrangling, no lbzip2 decompression, no 15 GB intermediate files.

---

## 2. Two-tier architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Consumer tier (root)                                           │
│                                                                 │
│  download_raw_data.py  ──►  snapshot_download(revision=tag)     │
│  run.sh                ──►  wrapper: venv + pip + invoke        │
│                                                                 │
│  Reads from: Hugging Face                                       │
│  Writes to:  /workspace/tmp/data/raw/                           │
└─────────────────────────────────────────────────────────────────┘
                            ▲
                            │  published artifact
                            │
┌─────────────────────────────────────────────────────────────────┐
│  Producer tier (archive/)                                       │
│                                                                 │
│  download_datasets.py  ──►  builds data/raw/ from upstream      │
│  upload_datasets.py    ──►  pushes to HF with git tag           │
│  requirements.txt      ──►  pinned deps                         │
│  run.sh                ──►  wrapper for download_datasets.py    │
│                                                                 │
│  Reads from: docketx/us-caselaw + CourtListener S3              │
│  Writes to:  data/raw/  +  intermediate/                        │
└─────────────────────────────────────────────────────────────────┘
```

**Why the split matters.** The producer tier is heavy (`~15 GB` of decompressed CSV, `~2.5 hours` at `--scale max`) and depends on outbound access to `storage.courtlistener.com`. The consumer tier is light (`426K edges → ~1.1 GB`), runs in **~2 minutes**, and works from any machine with `pip install huggingface_hub`. A reviewer never touches the producer tier; you run it once per scale and never again.

---

## 3. File inventory

### Consumer tier

| File | Role | Inputs | Outputs |
|---|---|---|---|
| **`download_raw_data.py`** | Pulls a published snapshot from Hugging Face | `DATASET_SIZE` (`min`/`medium`/`max`), `HF_USER_NAME`, optional `HF_TOKEN` | `data/raw/` with `corpus/`, `graph/`, `metadata/`, `eval_isolated/` |
| **`run.sh`** | Wrapper: activates venv, installs `huggingface_hub`, invokes `download_raw_data.py` | CLI args passed through | Same as above, plus pip install side effect |

### Producer tier — `archive/`

| File | Role | Inputs | Outputs |
|---|---|---|---|
| **`download_datasets.py`** | The main builder. Downloads upstream → converts bz2 to Parquet → filters to corpus → synthesizes eval → writes MANIFEST | `--scale`, `--dir`, `--force`, `--bulk-snapshot`, `--clean-stale`, `--clean-raw`, `--skip-clusters` | `corpus/opinions.parquet`, `graph/citations_filtered.parquet`, `metadata/clusters_filtered.parquet`, `eval_isolated/*`, `intermediate/*`, `README.md`, `LICENSE`, `DATA_SOURCES.md`, `MANIFEST.json` |
| **`upload_datasets.py`** | Pushes a built snapshot to Hugging Face and tags it (`v1.0-{scale}`) | `HF_USER_NAME`, `HF_TOKEN`, `DATASET_SIZE` | HF repo at `Athithya117/adaptive-graphrag-datasets`, commit + tag |
| **`requirements.txt`** | Pinned Python dependencies (`datasets`, `huggingface_hub`, `pandas`, `pyarrow`, `requests`, plus transitive) | — | Reproducible `pip install -r requirements.txt` |
| **`run.sh`** | Wrapper: activates `.venv`, installs requirements, invokes `download_datasets.py` | CLI args passed through | Same as `download_datasets.py` |

---

## 4. Pipeline flow

```
UPSTREAM                              PRODUCER                        ARTIFACT                    CONSUMER
─────────────────────────────────────────────────────────────────────────────────────────────────────────────
docketx/us-caselaw                    download_datasets.py            data/raw/                   download_raw_data.py
  (HF, streaming)              ──►    stream 5K–400K rows  ──►    corpus/opinions.parquet  ──►   snapshot_download(
                                       batch write                                ▲                 revision=v1.0-medium
CourtListener S3                       ▼                                          │
  citation-map-*.csv.bz2       ──►    lbzip2 → Parquet     ──►    graph/citations_filtered.parquet
  opinion-clusters-*.csv.bz2   ──►    lbzip2 → clean → Parquet ──► metadata/clusters_filtered.parquet
                                       ▼                                          │
feyninc/hojicha                ──►    load_dataset          ──►    eval_isolated/hojicha/*
                                       ▼                                          │
                                       synthesize 200 queries ──► eval_isolated/synthesized_graph_qa/
                                       ▼                                          │
                                       MANIFEST + README + LICENSE + DATA_SOURCES ─┘
                                                                                  │
                                       upload_datasets.py    ──►  HF repo + tag ──┘
```

The producer writes to `intermediate/` (77M-row citation Parquet, ~10M-row clean clusters Parquet) — these are **excluded from MANIFEST and upload** because they are 15 GB and rebuildable. Only the six filtered artifacts plus four metadata files (10 total) ship.

---

## 5. The data contract

### 5.1 Directory tree (identical at every scale)

```
data/raw/
├── README.md                         # HF dataset card
├── LICENSE                           # CC0 1.0
├── DATA_SOURCES.md                   # upstream attribution
├── MANIFEST.json                     # SHA-256 + row counts for every file
├── corpus/
│   └── opinions.parquet              # full-text opinions + metadata
├── graph/
│   └── citations_filtered.parquet    # (citing_opinion_id, cited_opinion_id, depth)
├── metadata/
│   └── clusters_filtered.parquet     # cluster-level metadata
└── eval_isolated/                    # NEVER indexed in production
    ├── hojicha/
    │   ├── corpus.parquet            # 194 contracts
    │   └── questions.parquet         # 1,568 chunking questions
    └── synthesized_graph_qa/
        └── queries.parquet           # 200 citation-lookup queries
```

### 5.2 Schemas

**`corpus/opinions.parquet`** — 15 columns

| Column | Type | Source field | Purpose |
|---|---|---|---|
| `id` | string | docketx `id` | Internal record ID |
| `cl_opinion_id` | int64 | `extra.cl_opinion_id` | **Join key to graph** |
| `cluster_id` | int64 | `extra.cluster_id` | **Join key to metadata** |
| `opinion_type` | string | `extra.opinion_type` | majority / dissent / concurrence |
| `text_kind` | string | `extra.text_kind` | original / combined |
| `title` | string | `title` | Case display name |
| `text` | string | `text` | **Full opinion body** — chunked for indexing |
| `citation` | string | `citation` | Bluebook citation (sparse key) |
| `court` | string | `court` | Court slug (metadata filter) |
| `date` | string | `date` | Decision date (temporal filter) |
| `jurisdiction` | string | `jurisdiction` | e.g., `us`, `us-ca9` |
| `doc_type` | string | `doc_type` | `opinion` / `order` |
| `license` / `source` / `retrieved_at` | string | provenance | Attribution + freshness |

**`graph/citations_filtered.parquet`** — 3 columns

| Column | Type | Semantics |
|---|---|---|
| `citing_opinion_id` | int64 | The source opinion (the one doing the citing) |
| `cited_opinion_id` | int64 | The target opinion (the one being cited) |
| `depth` | int32 | CourtListener's citation depth (1 = direct quote, >1 = inferred) |

Both endpoints are guaranteed to exist in `corpus/opinions.parquet.cl_opinion_id` — this is what "filtered to corpus" means.

**`metadata/clusters_filtered.parquet`** — 9 columns

| Column | Type | Purpose |
|---|---|---|
| `id` | int64 | Cluster ID — joins to `corpus.cluster_id` |
| `case_name` | string | Full case name |
| `case_name_short` | string | Abbreviated display |
| `date_filed` | string | Cluster decision date |
| `precedential_status` | string | `Published` / `Unpublished` — correctness filter |
| `citation_count` | int64 | Influence signal |
| `nature_of_suit` | string | Subject-matter filter |
| `judges` | string | Bench composition |
| `docket_id` | int64 | FK to `dockets` (not shipped; documented) |

**`eval_isolated/hojicha/*`** — chunking ablation set
- `corpus.parquet`: `title`, `text` (194 contracts)
- `questions.parquet`: `question`, `document_title`, `chunk-must-contain`, `question_type` (1,568 rows)

**`eval_isolated/synthesized_graph_qa/queries.parquet`** — 6 columns
- `query`, `eval_type`, `cited_opinion_id`, `ground_truth_citing_ids`, `ground_truth_count`, `derivation`
- Ground truth is **deterministic set membership** derived from `citations_filtered.parquet` — not hand-labeled

### 5.3 Join topology

```
corpus.opinions.cl_opinion_id  ◄──►  graph.citations.citing_opinion_id
corpus.opinions.cl_opinion_id  ◄──►  graph.citations.cited_opinion_id
corpus.opinions.cluster_id     ◄──►  metadata.clusters.id
metadata.clusters.docket_id    ──►   dockets.id          (not shipped)
dockets.court_id               ──►   courts.id           (court on corpus)
```

The chain is documented in `DATA_SOURCES.md` — the `docket_id → dockets → courts` link is **deliberately not materialized** because `corpus.court` and `corpus.jurisdiction` already carry the court metadata. Downloading `dockets` (larger than clusters) would add 2 hours of pipeline for zero retrieval benefit.

### 5.4 Scale presets

| Scale | Corpus rows | Citation edges | Clusters | Eval queries | Approx size | Build time |
|---|---|---|---|---|---|---|
| **`min`** | 5,000 | 1,951 | 4,780 | 200 | ~200 MB | ~3 min |
| **`medium`** | 100,000 | **426,462** | 92,157 | 200 | ~1.1 GB | ~10 min |
| **`max`** | 400,000 | ~1.7M (est.) | ~370,000 | 200 | ~4.5 GB | ~45 min |

**`min`** is a pipeline smoke test — the graph is too sparse (1,951 edges) for 2-hop traversal.
**`medium`** is the demo scale — 426K edges support multi-hop precedent chains and adaptive routing.
**`max`** is the portfolio claim — supports "~2M vectors" without inflating chunk counts.

---

## 6. Operational profile

| Stage | Peak RAM | Runtime (medium) | Idempotent | Disk intermediates |
|---|---|---|---|---|
| **Corpus** streaming | ~400 MB | ~3 min | Skip if `opinions.parquet` exists | None |
| **Citation bz2 → Parquet** | ~1.2 GB | ~4 min | Skip if `intermediate/citation_map.parquet` exists | 2.7 GB CSV (deleted after) |
| **Citation filter** | ~600 MB | ~3 min | Skip if `citations_filtered.parquet` exists | None |
| **Cluster bz2 → clean Parquet** | ~800 MB | ~8 min | Skip if `intermediate/clusters_clean.parquet` exists | 12 GB CSV (deleted after) |
| **Cluster filter** | ~300 MB | ~1 min | Skip if `clusters_filtered.parquet` exists | None |
| **Hojicha** | ~50 MB | ~5 s | Skip if exists | None |
| **Synthesize eval** | ~20 MB | ~1 s | Overwrites | None |

**All stages are skip-on-exist by default.** `--force` re-runs only the stages whose outputs exist. `--clean-raw` deletes `.csv` and `.csv.bz2` files after success (~15 GB reclaimed per scale). `--clean-stale` removes the old Indian-stack directories (`graph_overlay/`, `structured_metadata/`, `production_corpus/`, `eval_isolated/adalat-ai/`).

**Peak RAM is bounded at every scale** because:
- Corpus writes in 5,000-row batches
- Citations read in 500,000-row Parquet batches
- Clusters stream via `csv.reader`, one physical line at a time

Only `intermediate/` scales with the upstream file, and both intermediates are cleaned immediately after their filter stage.

---

## 7. Design decisions encoded in this structure

| Decision | Rationale |
|---|---|
| **Producer and consumer are separate directories** | The consumer must fit on a laptop; the producer must tolerate 15 GB intermediates. Mixing them forces the consumer to carry producer dependencies (`lbzip2`, `pyarrow`, multi-GB scratch space). |
| **Two wrapper `run.sh` scripts, one per tier** | Prevents consumers from accidentally triggering the producer, and gives each tier its own venv + `pip install` cycle. |
| **`intermediate/` is excluded from MANIFEST and upload** | The 77M-row citation Parquet is 15 GB and rebuildable. Shipping it would make the HF repo unusable and bloat the consumer download by 15×. |
| **`eval_isolated/` is a sibling of `corpus/`** | Forces downstream code to consciously choose whether it reads `corpus/` (indexed) or `eval_isolated/` (never indexed). Merging them invites the classic "evaluated on training data" bug. |
| **Six artifacts, one MANIFEST** | The MANIFEST covers exactly the files that ship — no silent uploads, no missing files, one SHA-256 per artifact. |
| **`court_id` explicitly absent from clusters** | Upstream schema places `court_id` on `dockets`. Rather than force a `dockets` download (2 hours, 0 retrieval benefit), the corpus carries `court` and `jurisdiction` directly. The join is documented, not materialized. |
| **`case_name` may contain HTML markup** | CourtListener bulk exports embed `<parties>`, `<docketnumber>` in unquoted CSV fields. The `int(row[0])` validation recovers 100% of target clusters despite 86% "skip" rate, because the skipped rows are continuation fragments of *other* clusters. Verified: 0 of 4,780 target clusters had truncated case names. |

---

## 8. What ships vs. what stays local

| Artifact | Ships to HF | Excluded |
|---|---|---|
| `corpus/opinions.parquet` | ✅ | — |
| `graph/citations_filtered.parquet` | ✅ | — |
| `metadata/clusters_filtered.parquet` | ✅ | — |
| `eval_isolated/hojicha/corpus.parquet` | ✅ | — |
| `eval_isolated/hojicha/questions.parquet` | ✅ | — |
| `eval_isolated/synthesized_graph_qa/queries.parquet` | ✅ | — |
| `README.md`, `LICENSE`, `DATA_SOURCES.md`, `MANIFEST.json` | ✅ | — |
| `intermediate/citation_map.parquet` | — | ✅ 15 GB, rebuildable |
| `intermediate/clusters_clean.parquet` | — | ✅ 3.5 GB, rebuildable |
| `graph/*.csv`, `graph/*.csv.bz2` | — | ✅ reclaimable via `--clean-raw` |
| `metadata/*.csv`, `metadata/*.csv.bz2` | — | ✅ reclaimable via `--clean-raw` |

Total published snapshot at `--scale medium`: **~1.1 GB across 10 files**. That is what `download_raw_data.py` fetches in ~2 minutes.
