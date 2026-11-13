# Legal GraphRAG

**Citation-aware retrieval over US case law** — built to close the 0.72 recall@5 ceiling from [RAG8s](https://github.com/Athithya-Sakthivel/RAG8s) via structure-aware chunking and native citation-graph traversal.

| | RAG8s | Legal GraphRAG |
|---|---|---|
| **Scope** | Full RAG platform (ingest → generate) | Retrieval only |
| **Corpus** | Heterogeneous internal docs | Structured US case law |
| **Chunking** | Flat 512-token windows | Parent–child, section-aligned |
| **Graph** | None | Native `CITES` edges, 1-hop |
| **Routing** | Fixed hybrid pipeline | Deterministic intent classifier |
| **Storage** | Qdrant + S3 + Valkey | PostgreSQL only |
| **Recall@5** | 0.72 | 0.91 macro |

*Corpora differ — directional comparison, not a controlled ablation.*

**No LLM, no generation, no containers. This project isolates one question: how much does structure-aware chunking and graph-native routing improve retrieval over a large structured corpus? Everything else is out of scope.**

---

## Results

150 stratified queries · 9,536-case corpus · 95% bootstrap CIs (n=1000) · CPU-only.

### Baseline

| Class | N | Recall@5 | Recall@10 | NDCG@10 | MRR |
|---|---|---|---|---|---|
| HOLDING | 50 | 0.880 [0.800, 0.960] | 0.880 [0.800, 0.960] | 0.865 | 0.860 |
| LOOKUP | 50 | 0.953 [0.892, 1.000] | 0.958 [0.897, 1.000] | 0.910 | 0.890 |
| VALIDITY | 30 | 0.907 [0.814, 0.983] | 0.950 [0.867, 1.000] | 0.929 | 0.938 |
| Negative control | 20 | — | — | — | — |
| **Macro** | | **0.913** | **0.929** | | |

Latency: p50 68ms · p95 237ms · mean 102ms.

### With `published_only` filter

| Class | Recall@5 | Recall@10 | Δ Recall@10 |
|---|---|---|---|
| HOLDING | 0.840 [0.740, 0.940] | 0.840 [0.740, 0.940] | −4.0% |
| LOOKUP | 0.953 [0.892, 1.000] | 0.958 [0.892, 1.000] | 0.0% |
| VALIDITY | 0.907 [0.814, 0.983] | 0.950 [0.867, 1.000] | 0.0% |
| **Macro** | | **0.916** | **−1.3%** |

HOLDING drops because some holdings appear in unpublished opinions that the filter correctly excludes. LOOKUP and VALIDITY are unchanged — citation hubs and overruling decisions are published.

### Failure modes

| Class | Breakdown |
|---|---|
| HOLDING | classification 18 · retrieval 1 |
| LOOKUP | resolution 1 · retrieval 1 |
| VALIDITY | retrieval 1 |
| Negative control | false_positive 18 |

The 18 HOLDING misclassifications are holding-text queries containing `v.` from cited precedents; the router sends them to LOOKUP, where the sparse arm still retrieves correctly. Negative controls return matches because there is no refusal mechanism at the retrieval layer.

---

## Architecture

Three query classes, each routed to the cheapest sufficient path.

| Class | Example | Path | Graph |
|---|---|---|---|
| HOLDING | "What did the court hold on qualified immunity?" | dense + sparse → RRF | no |
| LOOKUP | "Which cases cite Roe v. Wade?" | sparse + exact + 1-hop incoming → RRF | yes |
| VALIDITY | "Has Roe been overruled?" | 1-hop incoming → dense over citers → RRF | yes |

```
Query
  ↓
Intent classifier (regex, deterministic)
  extracts case_name · citation · opinion_id
  assigns HOLDING | LOOKUP | VALIDITY
  ↓
Retrieval arms
  dense   pgvector HNSW, halfvec(384), relaxed_order iterative scan
  sparse  tsvector + GIN, ts_rank_cd
  exact   citation / case-name equality, trigram fallback
  graph   1-hop CITES traversal, fan-out capped in SQL
  ↓
RRF fusion (per-class k values)
  ↓
Case diversity dedup
  ↓
Parent expansion (child chunk → full section)
  ↓
Ranked results + per-arm diagnostics
```

**Storage.** One PostgreSQL database — `cases`, `chunks`, `case_edges`. Vector search, lexical search, and graph traversal compose in a single SQL statement. No separate vector store, no graph database, no network round-trips between retrieval stages.

```sql
CREATE TABLE chunks (
    chunk_id   TEXT PRIMARY KEY,
    case_id    BIGINT REFERENCES cases(case_id),
    parent_id  TEXT,
    is_parent  BOOLEAN NOT NULL DEFAULT FALSE,
    section    TEXT,
    text_original TEXT NOT NULL,
    embedding  halfvec(384),
    tsv        TSVECTOR
);

CREATE TABLE case_edges (
    source_id BIGINT REFERENCES cases(case_id),
    target_id BIGINT REFERENCES cases(case_id),
    relation  TEXT NOT NULL,
    PRIMARY KEY (source_id, target_id, relation)
);
```

Foreign keys on both edge endpoints make dangling citations impossible.

---

## Chunking

Hierarchical: opinion → section → paragraph group → child chunk. Parents (1,200 tokens) are stored but not embedded; children (350 tokens) are the retrieval unit.

**Section detection.** Marker + separator hybrid across 15 section types, each with a priority used only as an RRF tie-breaker: `conclusion` 95, `judgment` 90, `holding` 88, `analysis` 85, `facts` 70, `syllabus` 40, `footnote` 20.

**Atomic spans.** Boundaries never fall inside case citations, statutory references, docket numbers, currency amounts, dates, or block quotations. A split citation is unrecoverable at retrieval time.

**Overlap.** Zero at structural boundaries. 32 tokens only when a single sentence exceeds the child cap.

**Breadcrumb.** Every child carries its legal identity in the embedded text:

```
[Roe v. Wade, 410 U.S. 113 (1973) · scotus · majority · § III.B]
The Court held that the right of privacy...
```

**Deterministic IDs.** `{opinion_id}:{opinion_type}:{section_index}:{child_index}` — re-ingestion is idempotent.

---

## Fusion

```
RRF(d) = Σ 1 / (k_arm + rank_arm(d))
```

| Class | dense k | sparse k | graph k | exact k |
|---|---|---|---|---|
| HOLDING | 60 | 60 | — | 15 |
| LOOKUP | — | 15 | 10 | 15 |
| VALIDITY | 60 | 20 | 10 | 15 |

Graph gets the lowest k because a direct `CITES` edge is deterministic evidence on a short, high-precision list — k=60 flattens the signal the arm exists to provide.

A sweep over k ∈ {5, 10, 20, 60} produced identical macro recall (0.926) at every value. At this recall level both arms agree on the answer, so fusion tuning is no longer the bottleneck.

---

## Metadata filtering

Applied inside the SQL `WHERE` clause, before the ANN scan — not post-filtered.

| Field | Index | Use |
|---|---|---|
| `court` | btree (court, date) | Jurisdiction scoping |
| `date` | btree (court, date) | Temporal scoping |
| `precedential_status` | btree | Precedential correctness |
| `case_id` | btree | Direct lookup / citer scoping |

`hnsw.iterative_scan = relaxed_order` with a materialized CTE re-sort. Without iterative scanning, filtered HNSW post-filters after the ANN scan and silently returns far fewer rows than requested.

---

## Quick start

**Requires:** Python 3.11+, PostgreSQL 15+ with `pgvector`, 8GB RAM.

```bash
git clone https://github.com/Athithya-Sakthivel/legal-graphrag.git
cd legal-graphrag

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

export DATABASE_URL="postgresql://user:pass@localhost:5432/legalrag"
export RETRIEVER_MODEL_DIR="src/indexing_pipeline/embedders"
```

```bash
# Download CourtListener corpus, citation graph, and cluster metadata
python data/archive/download_datasets.py --scale medium --dir data/

# Chunk, embed, load, index
python -m indexing_pipeline.cli all --corpus-rows 50000
```

```bash
# Retriever acceptance tests
bash src/retriever/test_e2e_locally.sh

# Build gold set and evaluate
bash src/evals/test_e2e_locally.sh --tier full --rebuild-gold
bash src/evals/test_e2e_locally.sh --tier full --published-only
bash src/evals/test_e2e_locally.sh --tier full --find-optimal-rrf-k
```

---

## Evaluation

**Gold set.** 150 queries generated deterministically from the live corpus, stratified proportionally by court, year bucket, and citation in-degree.

| Class | N | Query | Gold |
|---|---|---|---|
| HOLDING | 50 | Holding-chunk text, case name removed | seed case + chunk |
| LOOKUP | 50 | "Which cases cite {name}?" | all citing cases |
| VALIDITY | 30 | "Has {name} been overruled?" | overruling citers |
| Negative control | 20 | "Which cases cite {name}?" (indegree 0) | empty |

**Metrics.** Recall@5, Recall@10, NDCG@10, MRR — each with 95% bootstrap CIs (n=1000, fixed seed).

**Failure taxonomy.** Every query lands in exactly one bucket: `success`, `classification`, `resolution`, `retrieval`, `ranking`, `empty`, `false_positive`. Resolution failure is inferred from the absence of `graph_1hop` in the arm trace.

**Corpus drift guard.** Each gold row embeds a SHA-256 fingerprint of chunker versions, chunk IDs, case IDs, and edges. The runner refuses to score if the live corpus fingerprint diverges — this catches silent chunker drift, where retrieval is measured against gold whose chunk boundaries no longer exist.

---

## Limitations

1. **`CITES` edges only.** No `OVERRULES` or `DISTINGUISHES` — public citation data does not carry typed precedent edges. VALIDITY is graph-seeded and prose-verified, not typed-edge traversed.

2. **HOLDING pool is narrow.** Only ~1.3% of opinions have a detected `holding` section. The 50-query HOLDING eval samples 50 of 123 eligible cases — near-census of a section-detection-selected subset, not a random sample of holdings.

3. **Overruling gold is heuristic.** Substring match on `overrul|revers|vacat` stems. Produces false positives ("lower court reversed") and false negatives ("abrogated"). All gold rows are `verified=false`.

4. **No refusal mechanism.** Negative controls return lexical matches on the case's own text. Refusal belongs in a generation layer, which this project does not have.

5. **Single-node.** Optimized for correctness and latency at 10K–400K cases. No sharding, no replication, no horizontal scale path.

6. **Retrieval only.** Answer quality, faithfulness, and citation precision are unmeasured — there is no generation layer to measure.

---

## Non-goals

Legal advice · exhaustive precedent validation · multi-hop traversal · non-US jurisdictions · real-time ingestion · answer generation · Microsoft-style GraphRAG (no LLM entity extraction, no community detection, no hierarchical summaries).

---

## Docs

- [RAG design](docs/rag_design.md) — routing, arms, fusion, gates
- [Chunking](docs/chunking.md) — segmentation, atomic spans, validation
- [Datasets](docs/datasets.md) — schema, artifacts, scale presets
- [Evaluation](src/evals/README.md) — harness, metrics, failure taxonomy
