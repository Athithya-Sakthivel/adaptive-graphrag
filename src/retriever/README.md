# RAG Design — adaptive-graphrag

**US Case-Law Precedent Intelligence RAG.** Citation-aware retrieval over United States court opinions. Answers three query classes: holding, citation lookup, and precedent validity.

This is knowledge-graph RAG over an explicit citation graph. The graph is the CourtListener citation network (`CITES` edges). It is not LLM-extracted and does not carry typed precedent edges.

---

## 1. System Overview

A single-PostgreSQL retrieval system over US court opinions, combining three retrieval arms:

| Arm | Implementation | Source |
|---|---|---|
| Dense | `pgvector` HNSW, cosine, 384-dim | Opinion text chunks |
| Sparse | `tsvector` + GIN | Opinion text chunks |
| Graph | Adjacency table + recursive CTEs | `CITES` edges |

Fusion is Reciprocal Rank Fusion (RRF). Routing is a deterministic intent classifier. The LLM is invoked only for generation.

## 2. Scope Boundaries

- Graph carries `CITES` edges only. No `OVERRULES`, `DISTINGUISHES`, or `CONFLICTS_WITH`.
- US federal and state case law only.
- Single-hop graph traversal only. No multi-hop traversal.
- No Microsoft GraphRAG — no community detection, no hierarchical summaries, no LLM entity extraction.
- Not legal advice.

---

## 3. Corpus and Artifacts

Sourced from CourtListener bulk data (2026-06-30 snapshot).

| Role | Artifact | Description |
|---|---|---|
| Corpus | `corpus/opinions.parquet` | US court opinions with metadata |
| Graph | `graph/citations_filtered.parquet` | Citation edges (`CITES` only) |
| Metadata | `metadata/clusters_filtered.parquet` | Opinion cluster metadata |
| Eval — chunking | `eval_isolated/hojicha/` | Chunking quality benchmark |

---

## 4. Schema

### 4.1 Corpus — `corpus/opinions.parquet`

| Column | Type | Role |
|---|---|---|
| `cl_opinion_id` | int64 | Primary key |
| `cluster_id` | int64 | FK to metadata |
| `title` | string | Case display name |
| `text` | string | Chunking target |
| `citation` | string | Bluebook citation |
| `court` | string | Court slug |
| `date` | string | Decision date (ISO 8601) |
| `jurisdiction` | string | Jurisdiction code |
| `precedential_status` | string | Published / unpublished |

### 4.2 Graph — `graph/citations_filtered.parquet`

| Column | Type | Semantics |
|---|---|---|
| `citing_opinion_id` | int64 | Source opinion |
| `cited_opinion_id` | int64 | Target opinion |
| `depth` | int32 | CourtListener extraction heuristic |

Both endpoints guaranteed present in corpus. Only `CITES` edges exist.

### 4.3 Metadata — `metadata/clusters_filtered.parquet`

| Column | Type | Role |
|---|---|---|
| `id` | int64 | FK to `corpus.cluster_id` |
| `precedential_status` | string | Correctness filter |
| `date_filed` | string | Temporal filter |
| `nature_of_suit` | string | Subject-matter filter |
| `citation_count` | int64 | Influence signal |

---

## 5. Physical Schema

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
    precedential_status  TEXT,
    nature_of_suit       TEXT,
    citation_count       INT
);
CREATE INDEX idx_cases_court_date ON cases(court, date);
CREATE INDEX idx_cases_status     ON cases(precedential_status);

CREATE TABLE chunks (
    chunk_id   BIGSERIAL PRIMARY KEY,
    case_id    BIGINT REFERENCES cases(case_id),
    parent_id  BIGINT,
    section    TEXT,
    content    TEXT NOT NULL,
    embedding  halfvec(384),
    tsv        TSVECTOR
);
CREATE INDEX idx_chunks_embedding ON chunks USING hnsw (embedding halfvec_cosine_ops);
CREATE INDEX idx_chunks_tsv       ON chunks USING gin(tsv);
CREATE INDEX idx_chunks_case      ON chunks(case_id);

CREATE TABLE case_edges (
    source_id BIGINT REFERENCES cases(case_id),
    target_id BIGINT REFERENCES cases(case_id),
    relation  TEXT NOT NULL,
    PRIMARY KEY (source_id, target_id, relation)
);
CREATE INDEX idx_edges_source ON case_edges(source_id, relation);
CREATE INDEX idx_edges_target ON case_edges(target_id, relation);
```

`relation` is `CITES` for every edge. The column exists for future datasets with typed edges.

---

## 6. Query Classes and Routing

| Class | Example | Route | Graph |
|---|---|---|---|
| HOLDING | "What did the court hold on qualified immunity?" | Dense + sparse → RRF | No |
| LOOKUP | "Find cases citing Roe v. Wade" | Sparse + 1-hop graph | Yes (depth 1) |
| VALIDITY | "Is Roe still good law?" | Graph 1-hop → dense over citers | Yes (depth 1) |

### 6.1 HOLDING Routing

Dense retrieval (BGE-small-en-v1.5 embeddings) and sparse retrieval (PostgreSQL full-text search) run in parallel. Results fused via RRF. No graph access.

### 6.2 LOOKUP Routing

1. Exact/trigram match on case name or citation → `case_id`
2. 1-hop graph traversal: `SELECT source_id FROM case_edges WHERE target_id = ?`
3. Parent chunk expansion for retrieved cases
4. RRF fusion with sparse fallback

### 6.3 VALIDITY Routing

The graph carries `CITES` edges only. Typed precedent edges (`OVERRULES`) are proprietary. VALIDITY queries resolved by:

1. Sparse lookup of target opinion → `case_id`
2. 1-hop graph traversal to retrieve citing opinions
3. Dense retrieval over citing opinion text, filtered to retrieved `case_id`s
4. RRF fusion

The graph finds candidate overrulers. Dense retrieval confirms overruling language.

---

## 7. Chunking

Hierarchical: opinion body → section → paragraph → child chunk (320–460 tokens). Parent chunks (full sections) stored with `parent_id IS NULL`; child chunks reference their parent.

Section detection: `holding`, `facts`, `analysis`, `conclusion`, `procedural_history`, `dissent`, `concurrence`.

---

## 8. Metadata Filtering

Filters applied inside SQL `WHERE` clause (pre-retrieval, not post-filter).

| Field | Index | Use |
|---|---|---|
| `court` | btree | Jurisdiction scoping |
| `date` | btree | Temporal scoping |
| `precedential_status` | btree | Correctness filter |
| `nature_of_suit` | btree | Subject-matter scoping |

`precedential_status = 'Published'` is a correctness constraint. Unpublished opinions are not citable as precedent in most US jurisdictions.

---

## 9. Evaluation

Evaluated on 150 stratified queries across a 9,536-case corpus. All metrics include 95% bootstrap confidence intervals (n=1000).

### 9.1 Baseline (Unfiltered)

| Query Class | N | Recall@5 | Recall@10 | NDCG@10 | MRR | p95 Latency |
|---|---|---|---|---|---|---|
| HOLDING | 50 | 0.880 [0.800, 0.960] | 0.880 [0.800, 0.960] | 0.865 | 0.860 | 237ms |
| LOOKUP | 50 | 0.953 [0.892, 1.000] | 0.958 [0.897, 1.000] | 0.910 | 0.890 | — |
| VALIDITY | 30 | 0.907 [0.814, 0.983] | 0.950 [0.867, 1.000] | 0.929 | 0.938 | — |
| **Macro** | | | **0.929** | | | |

### 9.2 With `published_only` Filter

| Query Class | N | Recall@5 | Recall@10 | NDCG@10 | MRR | p95 Latency |
|---|---|---|---|---|---|---|
| HOLDING | 50 | 0.840 [0.740, 0.940] | 0.840 [0.740, 0.940] | 0.825 | 0.820 | 232ms |
| LOOKUP | 50 | 0.953 [0.892, 1.000] | 0.958 [0.892, 1.000] | 0.910 | 0.890 | — |
| VALIDITY | 30 | 0.907 [0.814, 0.983] | 0.950 [0.867, 1.000] | 0.929 | 0.938 | — |
| **Macro** | | | **0.916** | | | |

### 9.3 Filter Impact

| Query Class | Δ Recall@10 | Explanation |
|---|---|---|
| HOLDING | −4.0% | Some holdings in unpublished opinions excluded |
| LOOKUP | 0.0% | Landmark cases are published |
| VALIDITY | 0.0% | Overruling decisions are published |
| **Macro** | **−1.3%** | |

### 9.4 Failure Mode Analysis

| Query Class | Failure Modes |
|---|---|
| HOLDING | classification=18, retrieval=1 |
| LOOKUP | resolution=1, retrieval=1 |
| VALIDITY | retrieval=1 |
| negative_control | false_positive=18 |

HOLDING classification errors: 18 queries containing "v." in holding text misclassified as LOOKUP. Resolution failures minimal (1 each for LOOKUP/VALIDITY). Negative controls return false positives because retrieval layer has no refusal mechanism.

---

## 10. Known Limitations

1. **Graph carries `CITES` edges only.** VALIDITY queries are graph-seeded and prose-verified, not typed-edge traversed. Imposed by public citation data.

2. **No gold-answer RAG eval.** Retrieval metrics measurable; answer quality not measured. Requires ~50 hand-labeled holding questions with gold answers.

3. **HOLDING classification errors.** 18/50 HOLDING queries misclassified as LOOKUP when holding text contains case citations (e.g., "citing *Smith v. Jones*"). Router routes to LOOKUP, but sparse fallback still retrieves correctly.

4. **Negative control refusal.** Retrieval layer has no refusal mechanism. Negative controls (cases with zero citers) return BM25 matches on case text. Refusal belongs in generation layer.

5. **Section detection coverage.** Only ~1.3% of opinions have detected holding chunks. HOLDING eval is near-census of this narrow subset.

6. **Overruling heuristic.** VALIDITY gold uses substring matching on stems `overrul|revers|vacat`. Produces false positives ("lower court reversed") and false negatives ("abrogated").

---

## 11. Non-Goals

- Legal advice
- Exhaustive precedent validation
- Multi-hop graph traversal (single-hop only)
- Multi-jurisdiction (US only)
- Real-time ingestion
- Microsoft GraphRAG (no community detection, no LLM entity extraction)

---

## 12. Summary

Routing is deterministic: classifier selects retrieval arms per query class. Dense for holdings, sparse for citations, graph for precedent chains. The system is GraphRAG because LOOKUP and VALIDITY queries require citation-graph traversal. VALIDITY queries traverse `CITES` edges at depth 1 and rely on dense retrieval over citing opinions to confirm overruling; the public citation map does not carry typed precedent edges.
