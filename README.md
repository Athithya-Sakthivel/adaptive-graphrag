# Adaptive GraphRAG — Legal Precedent Intelligence

[RAG8s](https://github.com/Athithya-Sakthivel/RAG8s/tree/main) is a production-focused cloud RAG platform that achieved 0.95 groundedness but only 0.72 recall@5. The bottleneck wasn't the LLM or infrastructure — it was retrieval on unstructured, heterogeneous documents where flat chunking severed legal reasoning.
Legal GraphRAG is the on-premise answer to that specific problem. Built for structured legal corpora with explicit citation graphs, it achieves 0.93 macro recall@10 using structure-aware chunking, deterministic intent routing, and native PostgreSQL graph traversal — on a single node, CPU-only, no external APIs.


---

## Key Results

Evaluated on **150 stratified queries** across a **50,000-case corpus**. All metrics include **95% bootstrap confidence intervals** (n=1000).

### Baseline (Unfiltered)

| Query Class | N | Recall@5 | Recall@10 | NDCG@10 | MRR | p95 Latency |
|-------------|---|----------|-----------|---------|-----|-------------|
| **HOLDING** | 50 | 0.880 [0.800, 0.960] | 0.880 [0.800, 0.960] | 0.865 | 0.860 | 237ms |
| **LOOKUP** | 50 | 0.953 [0.892, 1.000] | 0.958 [0.897, 1.000] | 0.910 | 0.890 | — |
| **VALIDITY** | 30 | 0.907 [0.814, 0.983] | 0.950 [0.867, 1.000] | 0.929 | 0.938 | — |
| **Negative Control** | 20 | — | — | — | — | — |
| **Macro (pos.)** | | | **0.929** | | | |

### With `published_only` Filter

| Query Class | N | Recall@5 | Recall@10 | NDCG@10 | MRR | p95 Latency |
|-------------|---|----------|-----------|---------|-----|-------------|
| **HOLDING** | 50 | 0.840 [0.740, 0.940] | 0.840 [0.740, 0.940] | 0.825 | 0.820 | 232ms |
| **LOOKUP** | 50 | 0.953 [0.892, 1.000] | 0.958 [0.892, 1.000] | 0.910 | 0.890 | — |
| **VALIDITY** | 30 | 0.907 [0.814, 0.983] | 0.950 [0.867, 1.000] | 0.929 | 0.938 | — |
| **Negative Control** | 20 | — | — | — | — | — |
| **Macro (pos.)** | | | **0.916** | | | |

### Filter Impact

| Query Class | Δ Recall@10 | Explanation |
|-------------|-------------|-------------|
| HOLDING | −4.0% | Some holdings in unpublished opinions excluded |
| LOOKUP | 0.0% | Landmark cases are published |
| VALIDITY | 0.0% | Overruling decisions are published |
| **Macro** | **−1.3%** | |

### Failure Mode Analysis

| Query Class | Failure Modes |
|-------------|---------------|
| HOLDING | classification=18, retrieval=1 |
| LOOKUP | resolution=1, retrieval=1 |
| VALIDITY | retrieval=1 |
| negative_control | false_positive=18 |

HOLDING classification errors: 18 queries containing "v." in holding text misclassified as LOOKUP. Resolution failures minimal (1 each for LOOKUP/VALIDITY). Negative controls return false positives because retrieval layer has no refusal mechanism.

---

## Architecture

### Three Query Classes

| Class | Example | Retrieval Path | Graph Invoked |
|-------|---------|----------------|---------------|
| **HOLDING** | "What did the court hold on qualified immunity?" | Dense + sparse → RRF | No |
| **LOOKUP** | "Find cases citing Roe v. Wade" | Sparse + exact + 1-hop graph | Yes (depth 1) |
| **VALIDITY** | "Is Roe still good law?" | Graph 1-hop → dense over citers | Yes (depth 1) |

### Routing

Deterministic intent classifier routes queries to specialized retrieval pipelines:

```
Query
  │
  ▼
[1] Intent Parser (regex-based)
    extracts: case_name, citation, validity_keywords, holding_keywords
    assigns: HOLDING | LOOKUP | VALIDITY
  │
  ▼
[2] Route Selection
    HOLDING      → dense + sparse → RRF (one pass)
    LOOKUP       → sparse + exact + graph 1-hop incoming → RRF
    VALIDITY     → sparse + graph 1-hop incoming → dense over citers → RRF
  │
  ▼
[3] Retrieval Arms (parallel where independent)
    Dense:     pgvector HNSW, 384-dim embeddings, iterative_scan = strict_order
    Sparse:    PostgreSQL tsvector + GIN, ts_rank_cd
    Exact:     Citation/case-name btree match
    Graph:     1-hop traversal via case_edges table
  │
  ▼
[4] RRF Fusion (per-mode k values)
  │
  ▼
[5] Parent Expansion (child chunk → full section)
  │
  ▼
[6] Answer Generation (llama-server, static system prompt, abstention rule)
```

### Storage

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

CREATE TABLE chunks (
    chunk_id   BIGSERIAL PRIMARY KEY,
    case_id    BIGINT REFERENCES cases(case_id),
    parent_id  BIGINT,
    section    TEXT,
    content    TEXT NOT NULL,
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

---

## Quick Start

### Prerequisites

- Python 3.11+
- PostgreSQL 15+ with `pgvector` extension
- 8GB RAM minimum (16GB recommended for 50K corpus)

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/adaptive-graphrag.git
cd adaptive-graphrag

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export DATABASE_URL="postgresql://user:pass@localhost:5432/legalrag"
export RETRIEVER_MODEL_DIR="src/indexing_pipeline/embedders"
```

### Download Dataset

```bash
# Download CourtListener case law data (50K cases)
python data/archive/download_datasets.py --scale medium --dir data/

# This downloads:
# - corpus/opinions.parquet (50K opinions)
# - graph/citations_filtered.parquet (~2M citation edges)
# - metadata/clusters_filtered.parquet (opinion metadata)
```

### Index Corpus

```bash
# Run full indexing pipeline
python -m indexing_pipeline.cli all --corpus-rows 50000

# This creates:
# - cases table (50K rows)
# - chunks table (~250K rows, hierarchical chunking)
# - case_edges table (~2M citation edges)
# - HNSW index on embeddings
# - GIN index on tsvector
```

### Run Evaluation

```bash
# Build gold set (150 stratified queries)
bash src/evals/test_e2e_locally.sh --tier full --rebuild-gold

# Run baseline evaluation
bash src/evals/test_e2e_locally.sh --tier full

# Run with metadata filter
bash src/evals/test_e2e_locally.sh --tier full --published-only

# RRF k sweep (find optimal fusion parameter)
bash src/evals/test_e2e_locally.sh --tier full --find-optimal-rrf-k
```

### Run Retriever Tests

```bash
# End-to-end acceptance tests (5 test cases)
bash src/retriever/test_e2e_locally.sh
```

---

## Evaluation Methodology

### Gold Set Construction

150 queries stratified across three classes:
- **50 HOLDING:** Semantic search for case holdings (holding text as query)
- **50 LOOKUP:** Citation lookup (case name as query, gold = citing cases)
- **30 VALIDITY:** Overruling detection (case name as query, gold = overruling cases)
- **20 Negative Controls:** Cases with zero citers (measure false positives)

Queries generated deterministically from corpus with proportional stratification by court, year bucket, and citation in-degree.

### Metrics

- **Recall@K:** Fraction of gold case IDs present in top-K retrieved case IDs
- **NDCG@K:** Normalized discounted cumulative gain at rank K
- **MRR:** Mean reciprocal rank (1/rank of first gold case ID)
- **Bootstrap CIs:** 95% confidence intervals via bootstrap resampling (n=1000)

### Failure Mode Taxonomy

Every query categorized into exactly one failure mode:
- **success:** Gold case IDs present in top-K
- **classification:** Router assigned wrong query class
- **resolution:** LOOKUP/VALIDITY target case could not be resolved
- **retrieval:** Gold case IDs absent from all retrieved results
- **ranking:** Gold case IDs present but outside top-K
- **empty:** Answerable query returned zero hits
- **false_positive:** Negative control returned non-empty results

### Corpus Drift Detection

Gold set embeds SHA-256 fingerprint of corpus state at generation time. Eval runner refuses to score if live corpus fingerprint diverges from gold set's fingerprint. Prevents silent chunker drift.

---

## Chunking Strategy

Hierarchical: opinion body → section → paragraph → child chunk (320–460 tokens). Parent chunks (full sections) stored with `parent_id IS NULL`; child chunks reference their parent.

### Section Detection

15 section types detected via marker + separator hybrid:
- `conclusion` (priority 95)
- `judgment` / `disposition` (priority 90)
- `holding` (priority 88)
- `analysis` / `discussion` (priority 85)
- `facts` / `background` (priority 70)
- `syllabus` / `caption` (priority 40)
- `footnote` (priority 20, never promoted)

### Atomic Spans

Never split inside:
- Case citations (`410 U.S. 113`)
- Statutory citations (`42 U.S.C. § 1983`)
- Section references (`§ 1983`)
- Docket numbers (`No. 22-451`)
- Currency / amounts (`$85,000`)
- Dates (`March 27, 1985`)
- Block quotations

### Breadcrumb

Every child chunk begins with:
```
[{case_name_short}, {citation} ({year}) · {court} · {opinion_type} · § {section_path}]
```

Example:
```
[Roe v. Wade, 410 U.S. 113 (1973) · scotus · majority · § III.B]
The Court held that the right of privacy...
```

---

## Fusion — Reciprocal Rank Fusion (RRF)

### Formula

```
RRF(d) = Σ_arms  weight_arm / (k_arm + rank_arm(d))
```

### Per-Mode k Values

| Class | Dense k | Sparse k | Graph k | Exact k |
|-------|---------|----------|---------|---------|
| HOLDING | 60 | 60 | — | 15 |
| LOOKUP | — | 15 | 10 | 15 |
| VALIDITY | 60 | 20 | 10 | 15 |

### Rationale

- **Dense k=60:** Standard RRF default; symmetric treatment with sparse
- **Sparse k=15 (LOOKUP):** Sparse is exact for citations; trust its top ranks
- **Graph k=10:** Direct `CITES` edge is deterministic; low k concentrates signal at top
- **Exact k=15:** Exact match is high-confidence; trust its top ranks

---

## Metadata Filtering

Filters applied inside SQL `WHERE` clause (pre-retrieval, not post-filter).

| Field | Index | Use |
|-------|-------|-----|
| `court` | btree | Jurisdiction scoping |
| `date` | btree | Temporal scoping |
| `precedential_status` | btree | Correctness filter |
| `nature_of_suit` | btree | Subject-matter scoping |

`precedential_status = 'Published'` is a correctness constraint. Unpublished opinions are not citable as precedent in most US jurisdictions.

---

## Known Limitations

1. **Graph carries `CITES` edges only.** VALIDITY queries are graph-seeded and prose-verified, not typed-edge traversed. Imposed by public citation data.

2. **No gold-answer RAG eval.** Retrieval metrics measurable; answer quality not measured. Requires ~50 hand-labeled holding questions with gold answers.

3. **HOLDING classification errors.** 18/50 HOLDING queries misclassified as LOOKUP when holding text contains case citations (e.g., "citing *Smith v. Jones*"). Router routes to LOOKUP, but sparse fallback still retrieves correctly.

4. **Negative control refusal.** Retrieval layer has no refusal mechanism. Negative controls (cases with zero citers) return BM25 matches on case text. Refusal belongs in generation layer.

5. **Section detection coverage.** Only ~1.3% of opinions have detected holding chunks. HOLDING eval is near-census of this narrow subset.

6. **Overruling heuristic.** VALIDITY gold uses substring matching on stems `overrul|revers|vacat`. Produces false positives ("lower court reversed") and false negatives ("abrogated").

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Database | PostgreSQL 15+ |
| Vector Search | pgvector (HNSW, halfvec(384)) |
| Lexical Search | PostgreSQL tsvector + GIN |
| Embeddings | BGE-small-en-v1.5 (384-dim, OpenVINO) |
| Graph Traversal | Recursive CTEs on adjacency table |
| Chunking | Hierarchical structure-aware (1,200-token parents, 350-token children) |
| Evaluation | Bootstrap CIs, failure mode analysis, corpus drift detection |
| LLM | llama.cpp (LFM2.5-1.2B-Instruct-Q4_K_M) |

---

## Repository Structure

```sh
adaptive-graphrag/
├── data/
│   └── archive/                    # Dataset download scripts
├── src/
│   ├── indexing_pipeline/          # Chunking + embedding pipeline
│   │   ├── chunker.py              # Hierarchical structure-aware chunker
│   │   ├── embedder.py             # OpenVINO batch embedder
│   │   └── cli.py                  # Pipeline CLI
│   ├── retriever/                  # Retrieval system
│   │   ├── router.py               # Intent parser + route selection
│   │   ├── dense.py                # pgvector HNSW retrieval
│   │   ├── sparse.py               # tsvector retrieval
│   │   ├── exact.py                # Citation/case-name exact match
│   │   ├── graph.py                # 1-hop graph traversal
│   │   ├── fusion.py               # RRF fusion
│   │   ├── classify.py             # Intent classification
│   │   ├── config.py               # Retriever configuration
│   │   ├── filters.py              # Metadata filters
│   │   └── test_e2e_locally.sh     # End-to-end tests
│   └── evals/                      # Evaluation framework
│       ├── custom_eval_builder.py  # Gold set generator
│       ├── run_eval.py             # Eval orchestrator
│       ├── metrics.py              # Recall, NDCG, MRR, bootstrap CIs
│       ├── adapter.py              # Retriever adapter
│       ├── source.py               # Database candidate loader
│       ├── fingerprint.py          # Corpus drift detection
│       └── test_e2e_locally.sh     # Eval harness
├── results/                        # Evaluation reports (gitignored)
├── docs/                           # Design documents
│   ├── rag_design.md               # Retrieval architecture
│   ├── chunking.md                 # Chunking strategy
│   └── datasets.md                 # Dataset schema
├── requirements.txt
└── README.md
```

---

## Design Documents

- **[RAG Design](docs/rag_design.md):** Query classes, routing, retrieval arms, fusion, evaluation gates
- **[Chunking Strategy](docs/chunking.md):** Hierarchical chunking, section detection, atomic spans, validation gates
- **[Datasets](docs/datasets.md):** Schema, artifacts, scale presets

---

## Non-Goals

- Legal advice
- Exhaustive precedent validation
- Multi-hop graph traversal (single-hop only)
- Multi-jurisdiction (US only)
- Real-time ingestion
- Microsoft GraphRAG (no community detection, no LLM entity extraction)

---
