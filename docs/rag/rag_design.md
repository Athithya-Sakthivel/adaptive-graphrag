# RAG Design — adaptive-graphrag

**Indian Legal Precedent Intelligence RAG.** Citation-aware retrieval over
Indian court judgments. Answers three query classes: *what does this say*,
*is this still good law*, *how has this been applied*.

> This is **knowledge-graph RAG** over an explicit citation graph — **not**
> Microsoft GraphRAG (no LLM entity extraction, no community summaries).
> The graph is the citation network; it comes from the judgments themselves.

---

## 1. Stack

| Layer | Choice | Notes |
|---|---|---|
| Storage | **PostgreSQL only** — `pgvector` + `tsvector` + adjacency tables | One DB, one transaction boundary |
| Dense retrieval | `pgvector` HNSW | Cosine, 1536-dim |
| Sparse retrieval | `tsvector` + GIN | Statutory citations, case names, section numbers |
| Graph retrieval | Adjacency tables + `WITH RECURSIVE` | Bounded 2–3 hop traversal |
| Fusion | RRF in SQL | ~30 lines, no external service |
| Chunking | Hierarchical: section → paragraph → child (350–450 tok) | Overlap 40–80 tok |
| Routing | Deterministic intent parser + escalation threshold | LLM does not decide applicability |
| Filters | `court`, `year`, `bench`, `statute`, `precedential_status` | Indexed, applied inside query |

**Why one Postgres:** at ~2M vectors, dedicated vector/graph stores add
operational cost without measured benefit. Revisit at 10M+ or if p95 latency
on filtered queries exceeds budget.

---

## 2. Corpus

| Role | Dataset | Volume | Purpose |
|---|---|---|---|
| **Production corpus** | `overthelex/indian-court-decisions` (150K sample) | ~2.25M child vectors | Chunking, retrieval, filtering at scale |
| **Graph overlay** | `Viverun/jits-legal-dataset` (~10K) + `joyboseroy/inIRAC` (~500, typed edges) | ~150K vectors | Citation graph, `OVERRULES` / `DISTINGUISHES` |
| **Chunking ablation** | `feyninc/hojicha` (isolated) | 479 contracts, 1,982 questions | Fixed vs hierarchical Recall@5 |
| **RAG diagnostic** | `isaacus/legal-rag-qa` (isolated) | 138 QA pairs | Retrieval vs reasoning decomposition |
| **Indian gold eval** | `adalat-ai/Indian-Legal-Retrieval-Generation` (isolated) | 21 lawyer-verified pairs | Smoke test |

**Rule:** eval datasets never enter the production index. Isolated collections
deleted after recording metrics.

---

## 3. Schema (core tables)

```sql
CREATE TABLE cases (
    case_id     BIGSERIAL PRIMARY KEY,
    court_code  TEXT NOT NULL,
    year        INT  NOT NULL,
    bench       TEXT,
    precedential_status TEXT,
    full_text   TEXT NOT NULL,
    metadata    JSONB
);

CREATE TABLE chunks (
    chunk_id   BIGSERIAL PRIMARY KEY,
    case_id    BIGINT REFERENCES cases(case_id),
    parent_id  BIGINT,                    -- NULL for parents
    section    TEXT,
    content    TEXT NOT NULL,
    embedding  VECTOR(1536),
    tsv        TSVECTOR
);

CREATE INDEX idx_chunks_embedding ON chunks
    USING hnsw (embedding vector_cosine_ops);
CREATE INDEX idx_chunks_tsv ON chunks USING gin(tsv);
CREATE INDEX idx_chunks_case ON chunks(case_id);
CREATE INDEX idx_cases_court_year ON cases(court_code, year);

CREATE TABLE case_edges (
    source_id BIGINT REFERENCES cases(case_id),
    target_id BIGINT REFERENCES cases(case_id),
    relation  TEXT NOT NULL,              -- CITES | OVERRULES | DISTINGUISHES
    PRIMARY KEY (source_id, target_id, relation)
);
CREATE INDEX idx_edges_source ON case_edges(source_id, relation);
```

---

## 4. Query flow

```
Question
  │
  ▼
Intent parser (deterministic)
  - extracts: case_id, citation, statute, court, year, query_class
  - query_class ∈ {HOLDING, VALIDITY, APPLICATION, LOOKUP}
  │
  ▼
Route selection (cheapest arm that can answer correctly)
  │
  ├── HOLDING      → dense + sparse → RRF → rerank
  ├── LOOKUP       → sparse → graph expand (1 hop)
  ├── VALIDITY     → graph-first (2-hop cited_by, find OVERRULES edge) + dense for prose
  └── APPLICATION  → graph (2-hop) + metadata filter + dense + rerank
  │
  ▼
Fusion + rerank (Cohere or cross-encoder)
  │
  ▼
Parent expansion (child → parent context, bounded token budget)
  │
  ▼
ACL + freshness recheck on expanded parents
  │
  ▼
Generate (Foundry or LiteLLM) → claim-level citations → abstain if unsupported
```

**Escalation:** if first-pass Recall@K on a validation probe is below
threshold (TBD by benchmark), escalate to graph. Graph is not invoked for
every query — that is the *adaptive* claim.

---

## 5. Adaptive routing — the falsifiable policy

| Query class | Example | Arm(s) | Graph invoked? |
|---|---|---|---|
| HOLDING | "What did the court hold on X?" | dense + sparse | No |
| LOOKUP | "Cases citing 410 U.S. 113" | sparse + 1-hop expand | Yes (shallow) |
| VALIDITY | "Is case Y still good law?" | graph-first + dense prose | **Yes (mandatory)** |
| APPLICATION | "How has Z been applied since 2020?" | graph 2-hop + filter + dense | Yes (deep) |

If every query hit every arm, this would not be adaptive. The router is the
product; graph traversal is invoked only when relationships are load-bearing.

---

## 6. Metadata filtering

Filters are applied **inside** the query, not after retrieval.

| Field | Index | Used for |
|---|---|---|
| `court_code` | btree | Jurisdiction scoping |
| `year` | btree | Temporal scoping (`filed_after`) |
| `precedential_status` | btree | Correctness filter (unpublished ≠ binding) |
| `statute` | GIN (jsonb) | Statutory section lookup |
| `case_id` | btree | Direct lookup |

---

## 7. Evaluation

| Metric | Dataset | What it proves |
|---|---|---|
| Recall@5 / NDCG@10 | `legal-rag-qa` | Retrieval quality per arm |
| Retrieval vs reasoning failure split | `legal-rag-qa` | Where the pipeline breaks |
| Δ Recall@5 fixed vs hierarchical chunking | `hojicha` | Chunking mattered |
| Answer faithfulness | `adalat-ai` | Indian gold smoke test |
| p50/p95 latency per route | Production corpus | Adaptive routing cost |
| Citation accuracy | Hand-labeled sample | Graph traversal correctness |

**Four-arm ablation:** dense-only, sparse-only, graph-only, adaptive.
Same corpus, same questions, same LLM. Adaptive must win overall and each
arm must win its own query class.

---

## 8. Non-goals

- Not legal advice.
- Not exhaustive precedent validation (bounded 2–3 hops, not full Shepardizing).
- Not production CourtListener-scale (150K judgments, not 14.6M).
- Not multi-jurisdiction (`legal-rag-qa` is Australian; used to evaluate the
  system, not the domain).
- Not real-time RECAP upload (simulated locally with the RECAP contract rules).

---

## 9. Open questions / benchmark before claiming

- **Filtered HNSW latency** on `pgvector` at 2.25M with high-selectivity filters.
  pgvector 0.8+ iterative scans mitigate; must measure, not assume.
- **Recursive CTE depth** — where does 2-hop become 3-hop expensive?
  Benchmark before claiming "bounded traversal."
- **Sparse quality** — `tsvector` vs true BM25 for citation lookup.
  Measure Recall@K on citation queries.
- **Escalation threshold** — pick empirically, not by guess.

---

## 10. One-line defense

> Adaptive because the router picks the cheapest retrieval arm per query;
> GraphRAG because some legal questions cannot be answered without traversing
> the citation graph.
