# RAG Design — adaptive-graphrag

**US Case-Law Precedent Intelligence RAG.** Citation-aware retrieval over United States court opinions. Answers four query classes: holding, citation lookup, precedent application, and precedent validity.

This is knowledge-graph RAG over an explicit citation graph. The graph is the CourtListener citation network (`CITES` edges). It is not LLM-extracted and does not carry typed precedent edges.

---

## 1. System overview

A single-PostgreSQL retrieval system over 100K–400K US court opinions, combining four retrieval arms:

| Arm | Implementation | Source artifact |
|---|---|---|
| Dense | `pgvector` HNSW, cosine, 1536-dim | `corpus/opinions.parquet` → `text` |
| Sparse | `tsvector` + GIN | `corpus/opinions.parquet` → `text`, `citation`, `title` |
| Graph | Adjacency tables + `WITH RECURSIVE` | `graph/citations_filtered.parquet` |
| Filter | `btree` / `GIN` on cluster metadata | `metadata/clusters_filtered.parquet` |

Fusion is RRF in SQL. Routing is a deterministic intent parser with an escalation threshold. The LLM is invoked only for generation.

## 2. Scope boundaries

- Graph carries `CITES` edges only. No `OVERRULES`, `DISTINGUISHES`, or `CONFLICTS_WITH`.
- US federal and state case law only.
- Largest shipped scale: 400K corpus rows.
- No RECAP/PACER upload pipeline.
- No Microsoft GraphRAG — no community detection, no hierarchical summaries, no LLM entity extraction.
- Not legal advice.
- Not exhaustive precedent validation.

---

## 3. Corpus and artifacts

Published at `Athithya117/adaptive-graphrag-datasets`, tag `v1.0-{min|medium|max}`.

| Role | Artifact | Rows at `medium` |
|---|---|---|
| Corpus | `corpus/opinions.parquet` | 100,000 |
| Graph | `graph/citations_filtered.parquet` | 426,462 |
| Metadata | `metadata/clusters_filtered.parquet` | 92,157 |
| Eval — chunking | `eval_isolated/hojicha/` | 194 contracts, 1,568 questions |
| Eval — graph | `eval_isolated/synthesized_graph_qa/queries.parquet` | 200 queries |

`eval_isolated/` is never indexed. Evaluation collections are built, measured, and dropped.

Upstream sources: `docketx/us-caselaw` (corpus, mirrors CourtListener bulk 2026-06-30), CourtListener `citation-map-2026-06-30` (graph), CourtListener `opinion-clusters-2026-06-30` (metadata), `feyninc/hojicha` (chunking eval). Public domain or CC-BY-4.0 per `DATA_SOURCES.md`.

---

## 4. Schema

### 4.1 Corpus — `corpus/opinions.parquet`

| Column | Type | Role |
|---|---|---|
| `cl_opinion_id` | int64 | Primary key — every join originates here |
| `cluster_id` | int64 | FK to metadata; ~0.3% null |
| `title` | string | Case display name |
| `text` | string | Chunking target — the only field embedded/indexed |
| `citation` | string | Bluebook citation |
| `court` | string | Court slug (`ca9`, `scotus`, …) |
| `date` | string | Opinion decision date (ISO 8601) |
| `jurisdiction` | string | Jurisdiction code (`us`, `us-ca9`) |
| `opinion_type` | string | majority / dissent / concurrence |
| `text_kind` | string | original / combined / generated |
| `doc_type` | string | opinion / order |
| `id`, `license`, `source`, `retrieved_at` | string | Provenance |

`cl_opinion_id` is unique. `text` is non-null for every row. `court` and `jurisdiction` are populated directly; the `dockets` table is not shipped.

### 4.2 Graph — `graph/citations_filtered.parquet`

| Column | Type | Semantics |
|---|---|---|
| `citing_opinion_id` | int64 | Source opinion — FK to `corpus.cl_opinion_id` |
| `cited_opinion_id` | int64 | Target opinion — FK to `corpus.cl_opinion_id` |
| `depth` | int32 | CourtListener extraction heuristic; not a confidence score |

Both endpoints are guaranteed present in the corpus. Only `CITES` edges exist.

### 4.3 Metadata — `metadata/clusters_filtered.parquet`

| Column | Type | Role |
|---|---|---|
| `id` | int64 | FK to `corpus.cluster_id` |
| `precedential_status` | string | Correctness filter |
| `date_filed` | string | Temporal filter |
| `nature_of_suit` | string | Subject-matter filter |
| `citation_count` | int64 | Influence signal (corpus-independent) |
| `judges` | string | Bench composition |
| `case_name`, `case_name_short` | string | Display |
| `docket_id` | int64 | FK to unshipped `dockets` table |

`court_id` is not in this table. Court metadata lives on `corpus.court` / `corpus.jurisdiction`.

---

## 5. Physical schema

Single PostgreSQL database. No separate vector store, graph store, or document store.

```sql
CREATE TABLE cases (
    case_id              BIGINT PRIMARY KEY,
    cluster_id           BIGINT,
    court                TEXT,
    jurisdiction         TEXT,
    date                 DATE,
    title                TEXT,
    citation             TEXT,
    opinion_type         TEXT,
    precedential_status  TEXT,
    nature_of_suit       TEXT,
    citation_count       INT,
    judges               TEXT,
    metadata             JSONB
);
CREATE INDEX idx_cases_court_date ON cases(court, date);
CREATE INDEX idx_cases_status     ON cases(precedential_status);

CREATE TABLE chunks (
    chunk_id   BIGSERIAL PRIMARY KEY,
    case_id    BIGINT REFERENCES cases(case_id),
    parent_id  BIGINT,
    section    TEXT,
    content    TEXT NOT NULL,
    embedding  VECTOR(1536),
    tsv        TSVECTOR
);
CREATE INDEX idx_chunks_embedding ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX idx_chunks_tsv       ON chunks USING gin(tsv);
CREATE INDEX idx_chunks_case      ON chunks(case_id);

CREATE TABLE case_edges (
    source_id BIGINT REFERENCES cases(case_id),
    target_id BIGINT REFERENCES cases(case_id),
    relation  TEXT NOT NULL,
    depth     INT,
    PRIMARY KEY (source_id, target_id, relation)
);
CREATE INDEX idx_edges_source ON case_edges(source_id, relation);
CREATE INDEX idx_edges_target ON case_edges(target_id, relation);
```

`relation` is `CITES` for every shipped edge. The column exists for a future dataset with typed edges.

---

## 6. Query classes and routing

| Class | Example | Route | Graph invoked |
|---|---|---|---|
| HOLDING | "What did the court hold on qualified immunity?" | dense + sparse → RRF → rerank | No |
| LOOKUP | "Find cases citing Roe v. Wade" | sparse + 1-hop graph expand | Yes (depth 1) |
| APPLICATION | "How has the undue-burden standard been applied?" | graph 2-hop + metadata filter + dense + rerank | Yes (depth 2) |
| VALIDITY | "Is Roe still good law?" | sparse → graph 1-hop `cited_by` → dense over citing opinions | Yes (depth 1) |

### 6.1 VALIDITY routing

The graph carries `CITES` edges only. Typed precedent edges (`OVERRULES`) are proprietary. VALIDITY queries are resolved by:

1. Sparse lookup of the target opinion → `cl_opinion_id`
2. Graph 1-hop `cited_by` to retrieve citing opinions, bounded by `citation_count`
3. Dense retrieval over the text of those citing opinions
4. LLM synthesis citing the overruling language

The graph finds candidate overrulers. The prose confirms overruling.

### 6.2 Adaptive routing

The router selects the cheapest arm(s) per query class:

- HOLDING queries do not access `case_edges`.
- LOOKUP queries access `case_edges` at depth 1.
- APPLICATION queries access `case_edges` at depth 2 with a metadata prefilter.
- VALIDITY queries access `case_edges` at depth 1 and rely on dense retrieval for the decision.

Graph load per route is observable via `idx_edges_source` hit counts.

### 6.3 Escalation

First-pass dense+sparse retrieval returns top-K. If top-K recall on the validation probe falls below a benchmarked threshold, the router escalates to graph. The threshold is measured against `synthesized_graph_qa`.

---

## 7. Chunking

Hierarchical: opinion body → section → paragraph → child chunk (350–450 tokens, 40–80 token overlap). Parent chunks (1,000–2,000 tokens) are stored with `parent_id IS NULL`; child chunks reference their parent.

Chunking ablation runs on `eval_isolated/hojicha/`:

- Baseline: fixed 512-token chunks
- Treatment: hierarchical chunker
- Metric: Recall@5 on `chunk-must-contain` ground truth
- Collection is isolated and dropped after measurement

`hojicha` is contract text (CUAD-derived). The property measured — whether a required passage survives chunking — is independent of document type. The ablation measures chunker quality, not legal RAG quality.

---

## 8. Metadata filtering

Filters are applied inside the SQL query.

| Field | Index | Use |
|---|---|---|
| `court` | btree (composite with `date`) | Jurisdiction scoping |
| `date` | btree (composite with `court`) | Temporal scoping |
| `precedential_status` | btree | Correctness filter |
| `nature_of_suit` | btree | Subject-matter scoping |
| `citation` | GIN on `tsvector` | Citation-string lookup |
| `case_id` | btree | Direct lookup |

`precedential_status = 'Published'` is a correctness constraint. Unpublished opinions are not citable as precedent in most US jurisdictions. The metadata table is the only reliable source for this field.

---

## 9. Evaluation

| Metric | Dataset | Measures |
|---|---|---|
| Δ Recall@5 (fixed vs hierarchical chunking) | `hojicha` | Chunker quality |
| Recall@10, NDCG@10 on citation lookup | `synthesized_graph_qa` | Graph arm correctness |
| Citation precision | `synthesized_graph_qa` + `corpus` | Whether returned cases carry the claimed citation |
| Router accuracy | Both | Whether the router selects the correct arm |
| p50 / p95 latency per route | Production corpus | Cost of adaptive routing |

No gold-answer holding-question set is shipped. Answer faithfulness and end-to-end correctness cannot be measured with the current artifacts. The four-arm retrieval ablation (dense-only, sparse-only, graph-only, adaptive) can be run on retrieval metrics but not on answer quality.

Closing this gap requires ~50 hand-labeled holding questions with gold answers.

---

## 10. Scale presets

| Scale | Corpus rows | Graph edges | Chunks (est.) | Vectors (dense + sparse) | Build time |
|---|---|---|---|---|---|
| `min` | 5,000 | 1,951 | ~25K | ~50K | ~3 min |
| `medium` | 100,000 | 426,462 | ~500K | ~1M | ~10 min |
| `max` | 400,000 | ~1.7M | ~2M | ~4M | ~45 min |

At `medium`, ~4.26 citations per opinion produce multiple valid intermediate nodes for 2-hop traversal, which is the precondition for adaptive routing.

At `min`, 1,951 edges across 5,000 opinions produce almost no 2-hop paths. Graph-vs-dense comparisons at `min` measure noise.

The "~2M vectors" claim corresponds to `--scale max` (~400K corpus rows → ~2M chunks). At `medium`, the index is ~500K chunks (~1M vector entries across dense + sparse).

---

## 11. Known limitations

1. Graph carries `CITES` edges only. VALIDITY queries are graph-seeded and prose-verified, not typed-edge traversed. This limitation is imposed by the public citation data.
2. No gold-answer RAG eval is shipped. Retrieval metrics are measurable; answer quality is not.
3. `medium` scale produces ~1M vector entries, not 2M. The 2M claim requires `max`.
4. ~0.3% of opinions lack `cluster_id`. They are indexed for retrieval and excluded from metadata-filtered queries.
5. `case_name` may contain residual HTML from fragmented upstream CSV rows. Display-only; join keys and citation matching are unaffected.
6. `citation_count` reflects CourtListener's full-corpus count, not in-degree within the shipped graph.
7. `synthesized_graph_qa` contains 200 queries. The synthesis rule is a single `value_counts()`; extending to 1,000 queries is a one-line change.

---

## 12. Benchmarks required before further claims

| Claim | Required measurement |
|---|---|
| Filtered HNSW at `max` scale meets p95 latency budget | `pgvector` 0.8+ iterative scans on the actual corpus with high-selectivity filters |
| Bounded 2-hop traversal | Depth at which recursive CTE cost becomes unacceptable on the actual graph |
| Sparse retrieval is competitive for citation lookup | Recall@K on citation queries vs dense |
| Adaptive routing improves over single-arm baselines | Four-arm ablation on identical corpus, questions, and LLM |
| Answer quality is acceptable | Requires 50 gold-answer holding questions (not shipped) |

Priority:

1. Add ~50 hand-labeled holding questions with gold answers.
2. Benchmark filtered HNSW at `max` scale with p50/p95 reported.
3. Extend `synthesized_graph_qa` to 1,000 queries.

---

## 13. Non-goals

- Legal advice
- Exhaustive precedent validation (bounded 2–3 hop traversal)
- Production CourtListener scale (400K opinions at `max`, not 14.6M)
- Multi-jurisdiction (US only)
- Real-time upload (no PACER/RECAP ingestion)
- Microsoft GraphRAG (no community detection, no LLM entity extraction)

---

## 14. Summary

Routing is adaptive: the router selects the cheapest arm per query — dense for holdings, sparse for citations, graph for precedent chains. The system is GraphRAG because LOOKUP and APPLICATION queries require citation-graph traversal. VALIDITY queries traverse `CITES` edges at depth 1 and rely on dense retrieval over citing opinions to confirm overruling; the public citation map does not carry typed precedent edges.
