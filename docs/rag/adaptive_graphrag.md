# Adaptive GraphRAG — End-to-End Retrieval Plan

**Status:** Finalized
**Scope:** US case-law opinions, 1–2M vectors, PostgreSQL + pgvector + recursive-CTE citation graph
**Depends on:** [datasets.md](./datasets.md), [chunking.md](./chunking.md)

---

## 1. What This Document Specifies

Authoritative retrieval plan covering:

1. Query classification and routing
2. Dense, sparse, exact-match, and graph retrieval arms
3. Two-pass RRF fusion with per-mode and per-arm rank constants
4. Parent-context expansion
5. CPU-bounded generation
6. Evaluation gates and kill criteria

**Normative language.** `MUST` = a gate that blocks release. `SHOULD` = a default that may be overridden only with a recorded ablation result.

---

## 2. Query Classes

| Class | Example | Retrieval Path | Graph Invoked |
|---|---|---|---|
| **HOLDING** | "What did the court hold on qualified immunity?" | Dense + sparse → RRF (one pass) | No |
| **LOOKUP** | "Which cases cite Roe v. Wade?" | Sparse + exact + graph 1-hop incoming | Yes, 1 hop |
| **VALIDITY** | "Has Roe v. Wade been overruled?" | Sparse + graph 1-hop incoming → dense over citers | Yes, 1 hop |
| **APPLICATION** | "How has Casey's undue-burden standard been applied since 2022?" | Dense + sparse → seeds → graph 2-hop outgoing → metadata filter | Yes, 2 hops (gated) |

**Classification is deterministic**, based on parsed entities and query verbs:

```python
def classify(query):
    has_case = extract_case_reference(query) is not None
    if not has_case:
        return "HOLDING"
    if any(t in query for t in ["overrule", "overruled", "good law", "still valid"]):
        return "VALIDITY"
    if any(t in query for t in ["cite", "cited", "citing", "cited by"]):
        return "LOOKUP"
    if any(t in query for t in ["applied", "since", "evolution", "chain", "history"]):
        return "APPLICATION"
    return "HOLDING"   # default: cheapest sufficient path
```

**Default is HOLDING.** A query that cannot be confidently classified does not pay the graph cost.

---

## 3. Pipeline

```
Question
   │
   ▼
[1] Intent parser (deterministic)
    extracts: case_name, citation, statute, court, year
    assigns: HOLDING | LOOKUP | VALIDITY | APPLICATION
   │
   ▼
[2] Route selection
    HOLDING      → arms D, S, E              (one RRF pass)
    LOOKUP       → arms S, E, G1             (one RRF pass)
    VALIDITY     → arms S, G1 → D over citers (one RRF pass)
    APPLICATION  → arms D, S → seeds → G2 → D (two RRF passes)
   │
   ▼
[3] Retrieval arms (parallel where independent)
    D  dense     pgvector HNSW top-50, iterative_scan = strict_order
    S  sparse    tsvector top-50 (or pg_textsearch BM25 if available)
    E  exact     citation / statute btree match, cap 20
    G1 graph 1-hop incoming or outgoing, cap 50
    G2 graph 2-hop outgoing, cap 30 per hop, deduped, cap 200 total
   │
   ▼
[4] RRF pass 1 (D + S + E)  → top-30 seeds
   │
   ▼
[5] Graph expansion (only if class requires it)
    G1: from seed case_ids, one hop
    G2: from top-20 seed case_ids, two hops, capped
   │
   ▼
[6] RRF pass 2 (seeds + graph candidates) — weighted, per-arm k
   │
   ▼
[7] Dedup by chunk_id
   │
   ▼
[8] Parent expansion — child → parent section, cap 1,500–2,000 tokens
   │
   ▼
[9] Metadata recheck on expanded parents
   │
   ▼
[10] Generator (llama-server, stream, max_tokens 500, static system prompt)
   │
   ▼
[11] Citation validation → answer, or abstain
```

---

## 4. Dense Retrieval

### 4.1 Index

```sql
CREATE INDEX CONCURRENTLY idx_chunks_embedding
  ON chunks USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);
```

### 4.2 Query with Iterative Scans

```sql
BEGIN;
SET LOCAL hnsw.ef_search = 100;
SET LOCAL hnsw.iterative_scan = strict_order;

SELECT c.chunk_id, c.content, c.embedding <=> $1 AS dist
FROM chunks c
JOIN cases ca ON ca.case_id = c.case_id
WHERE ca.court = $2
  AND ca.date >= $3
  AND ca.precedential_status = 'Published'
ORDER BY c.embedding <=> $1
LIMIT 50;
COMMIT;
```

**`iterative_scan` is mandatory.** Without it, filtered HNSW queries post-filter after the ANN scan and return far fewer rows than requested.

### 4.3 The Filtered-Recall Gate

Before release, measure **Recall@50 against exact kNN** on the same filtered subset at three selectivity levels: 1%, 10%, 50%.

| Selectivity | Required Filtered Recall |
|---|---|
| 1% | ≥ 95% of exact recall |
| 10% | ≥ 97% of exact recall |
| 50% | ≥ 99% of exact recall |

If the 1% case fails, add a **partial HNSW index** on the dominant filter value or reduce filter selectivity. Do not ship a filtered query that has not passed this gate.

---

## 5. Sparse Retrieval

### 5.1 Two Implementations, One Role

| Implementation | When | Notes |
|---|---|---|
| **`pg_textsearch` (BM25)** | If available on the deployment | Preferred; real IDF and length normalisation |
| **`tsvector` + `ts_rank_cd`** | Always available | **Not BM25** — no IDF, no term-frequency saturation. Acceptable as a lexical ranker |

```sql
SELECT chunk_id, content,
       ts_rank_cd(tsv, websearch_to_tsquery('english', $1)) AS lex
FROM chunks
WHERE tsv @@ websearch_to_tsquery('english', $1)
ORDER BY lex DESC
LIMIT 50;
```

**Terminology discipline.** The plan MUST NOT describe `ts_rank_cd` as BM25.

---

## 6. Exact-Match Retrieval

```sql
SELECT case_id, citation, title
FROM cases
WHERE citation = $1        -- e.g. '410 U.S. 113'
LIMIT 20;
```

**Exact match is a separate arm, not a filter on sparse.** BM25 tokenizes `410 U.S. 113` into four tokens and scores it as a bag of words; a btree equality lookup returns the exact case.

---

## 7. Graph Retrieval

### 7.1 Storage

```sql
CREATE TABLE case_edges (
  source_id BIGINT NOT NULL REFERENCES cases(case_id),
  target_id BIGINT NOT NULL REFERENCES cases(case_id),
  relation  TEXT   NOT NULL,
  depth     INT,
  PRIMARY KEY (source_id, target_id, relation)
);
CREATE INDEX idx_edges_source ON case_edges (source_id, relation);
CREATE INDEX idx_edges_target ON case_edges (target_id, relation);
```

**Referential integrity is enforced at the schema level.** Both endpoints are foreign keys into `cases`.

### 7.2 Why Recursive CTE and Not Apache AGE

| Depth | Recursive CTE p50 | AGE Variable-Length Cypher p50 |
|---|---|---|
| 1 hop | **0.34 ms** | 33.7 ms |
| 2 hops | **0.42 ms** | 52.4 ms |
| 3 hops | **0.95 ms** | 132.1 ms (p95 670 ms) |

AGE and pgvector do not compose in one query. With AGE, Cypher results are wrapped in `agtype` and vector operators want plain columns, so implementations run two round-trips.

### 7.3 1-Hop (LOOKUP, VALIDITY)

Outgoing — what does this case cite:

```sql
SELECT target_id FROM case_edges
WHERE source_id = $1 AND relation = 'CITES'
LIMIT 50;
```

Incoming — what cites this case:

```sql
SELECT source_id FROM case_edges
WHERE target_id = $1 AND relation = 'CITES'
LIMIT 100;
```

**Cap at the query, not in application code.**

### 7.4 2-Hop (APPLICATION Only)

```sql
WITH RECURSIVE walk AS (
  SELECT target_id, 1 AS hop
  FROM case_edges
  WHERE source_id = $1 AND relation = 'CITES'
  LIMIT 30                                    -- fan-out cap at hop 1
  UNION
  SELECT e.target_id, w.hop + 1
  FROM walk w
  JOIN case_edges e ON e.source_id = w.target_id
  WHERE e.relation = 'CITES' AND w.hop < 2
)
SELECT DISTINCT target_id, MIN(hop) AS hop
FROM walk
GROUP BY target_id
LIMIT 200;
```

**Three mandatory constraints:**

1. **Fan-out cap of 30 at hop 1.**
2. **`UNION` (not `UNION ALL`) for cycle safety.**
3. **`DISTINCT` + `MIN(hop)`.**

**2-hop is gated, not automatic.** Enabled only for APPLICATION queries, and only if the ablation in §12 shows it beats 1-hop on multi-hop Recall@10 by ≥5 points.

---

## 8. Fusion — Two-Pass RRF

### 8.1 Formula

```
RRF(d) = Σ_arms  weight_arm / (k_arm + rank_arm(d))
```

### 8.2 Pass 1 — Seed Selection (Dense + Sparse + Exact)

**Per-mode `k` for pass 1:**

| Class | Dense `k` | Sparse `k` | Rationale |
|---|---|---|---|
| HOLDING | 60 | 60 | Both arms are approximate; symmetric treatment |
| LOOKUP | 60 | 15 | Sparse is exact for citations; trust its top ranks |
| VALIDITY | 60 | 20 | Sparse resolves the case name |
| APPLICATION | 60 | 20 | Same as VALIDITY |

**`k=60` is a starting convention.** `k ∈ {10, 30, 60, 120}` MUST be swept on this corpus before release (§12).

### 8.3 Pass 2 — Graph Merge (Weighted)

| Arm | Weight | `k` | Rationale |
|---|---|---|---|
| Hybrid seeds | 0.85 | 60 | Already fused; rank 1 is the consensus result |
| Graph | **0.15** | **10** | A direct `CITES` edge is deterministic; low `k` concentrates its signal at the top |

**Why the graph arm gets the lowest `k`.** With `k=10`, rank 1 scores `0.091` and rank 10 scores `0.050` — an 82% difference. On a shallow, high-precision list, `k=60` flattens away the very signal the graph arm exists to provide.

### 8.4 One Pass or Two — The Rule

| Class | RRF Passes | Arms Fused |
|---|---|---|
| HOLDING | **1** | Dense + sparse (+ exact) |
| LOOKUP, VALIDITY | **1** | Sparse + exact + graph 1-hop |
| APPLICATION | **2** | Pass 1: dense + sparse → seeds. Pass 2: seeds + graph 2-hop |

**Never fuse graph and dense in the same pass.** The graph arm has a hard cut at its fan-out cap; dense and sparse have long tails.

---

## 9. Parent Expansion and Context Budget

After pass 2, deduplicate by `chunk_id`, then expand each selected child to its parent section.

| Parameter | Value |
|---|---|
| Expansion | child → parent (`parent_id`) |
| Context cap | **1,500 tokens** (default), 2,000 absolute maximum |
| Selection | top-N by fused score until the cap is reached |
| Re-check | re-validate `court`, `date`, `precedential_status` on the expanded parent |

**Why 1,500 and not 4,000.** The context cap is a **latency contract**, not a quality preference, and it must be set from a measured TTFT curve (§12, gate L2).

---

## 10. Generation

### 10.1 Server Configuration

```bash
./llama-server \
  -m /workspace/models/llm/LFM2.5-1.2B-Instruct-Q4_K_M.gguf \
  -c 8192 \
  -t 4 -tb 4 \
  --jinja \
  --port 8080
```

### 10.2 Request Contract

```json
{
  "model": "lfm2.5-1.2b-instruct",
  "messages": [
    {"role": "system", "content": "<STATIC_PROMPT>"},
    {"role": "user",   "content": "<question>\n\nEVIDENCE:\n<passages>"}
  ],
  "max_tokens": 500,
  "temperature": 0,
  "stream": true
}
```

**The system prompt MUST be static.** A prompt that changes per request loses KV-cache prefix reuse.

### 10.3 Abstention

The prompt contains a hard rule:

> If the evidence does not contain a direct answer, respond exactly `Insufficient evidence` and do not speculate.

---

## 11. Evaluation — The Release Gates

### 11.1 Metric Set

| Layer | Metric | Target |
|---|---|---|
| Retrieval | Recall@10, Recall@50 | Baseline-relative; no fixed absolute |
| Ranking | nDCG@10 | Baseline-relative |
| Graph | Path validity, entity-resolution precision | 100% valid paths (FK-enforced) |
| Answer | Citation precision | **≥ 90%** for decision-support workloads |
| Answer | Faithfulness | No more than 2% regression vs baseline |
| Safety | Abstention accuracy on negative queries | Measured, reported |
| Operations | p50/p95 latency, filtered recall | Measured, reported |

**Evaluation set.** 200–500 independently reviewed questions. Recommended stratification: **20% simple lookup, 40% multi-hop, 20% aggregation/comparison, 20% adversarial**.

### 11.2 Ablation — Four Configurations

| Config | Components | Must Beat |
|---|---|---|
| **A** | Dense only | — (floor) |
| **B** | Dense + sparse + exact + RRF (one pass) | A on Recall@10 and nDCG@10 by ≥3 points |
| **C** | B + graph 1-hop (per-mode k) | B on LOOKUP/VALIDITY by ≥5 points |
| **D** | C + graph 2-hop (APPLICATION only) | C on multi-hop Recall@10 by ≥5 points |

**Additional sweeps required:**

| Sweep | Values | Decision |
|---|---|---|
| **RRF k** | 10, 30, 60, 120 | Adopt the sweep winner; document it |
| **Graph k** | 5, 10, 20 | Adopt the sweep winner for the graph arm |
| **Context cap** | 1,000 / 1,500 / 2,000 tokens | Adopt the largest cap whose TTFT meets target |

### 11.3 Kill Criteria

| Condition | Action |
|---|---|
| B does not beat A by ≥3 points Recall@10 | Fusion is not helping; inspect sparse coverage before proceeding |
| C does not beat B on LOOKUP/VALIDITY by ≥5 points | Graph 1-hop is decorative; drop it from those classes |
| D does not beat C on multi-hop by ≥5 points | **Drop 2-hop and remove the APPLICATION class from the README claim** |
| Filtered recall at 1% selectivity < 95% of exact | Do not ship filtered queries; add partial index or reduce selectivity |
| 1,500-token TTFT > 10 s on target hardware | Lower context cap to 1,000 tokens |
| Citation precision < 90% | Do not claim decision-support suitability |

**The kill criteria are the release contract.** A component that does not clear its gate is removed, not defended.

---

## 12. Failure Modes, Ranked

| Rank | Failure | Detection | Mitigation |
|---|---|---|---|
| **1** | Filtered HNSW silent recall loss | Recall@50 vs exact kNN at 1% selectivity | `iterative_scan = strict_order`; partial index if the gate fails |
| **2** | Router misclassification | Per-class route accuracy on the eval set | Conservative default to HOLDING; rule-based parser; log every route |
| **3** | Graph fan-out explosion | Candidate count after pass 1 expansion | Hard cap 30 per hop; `DISTINCT` + `MIN(hop)` |
| **4** | Prefill timeout on CPU | TTFT at 1,500 tokens | Context cap from measured curve; static system prompt for KV reuse |
| **5** | Dangling graph edges | Empty traversal from a known seed | FK constraints on `case_edges`; ingestion filter |
| **6** | Stale embeddings | Semantically wrong retrieval after content update | Cache keyed by `(chunk_id, chunker_version, embedding_model)`; re-embed on change |
| **7** | Unsupported answer | Citation precision on generated text | Abstention rule; citation validator; ≥90% precision gate |

---

## 13. Scale Checkpoints

| Scale | Corpus Rows | Vectors (est.) | HNSW Working Set (est.) | Minimum RAM |
|---|---|---|---|---|
| `min` | 5,000 | ~25K | < 1 GB | 4 GB — smoke test only |
| `medium` | 100,000 | ~500K | ~4 GB | 8 GB |
| `max` | 400,000 | ~2M | **~15–50 GB** | **32 GB minimum, 64 GB preferred** |

The HNSW working-set estimate uses the measured **20–25 KB per vector** figure for real deployments with graph metadata, not the raw 6 KB float32 payload.

---

## 14. Explicit Non-Goals

| Not Implemented | Why |
|---|---|
| **Cross-encoder reranker** | Evidence in the legal domain favours no-rerank on strong first-stage retrieval; CPU cost is disproportionate |
| **Apache AGE** | 1–3 hop traversal is 1.4–47× slower than recursive CTE, cannot compose with pgvector in one query |
| **Microsoft-style GraphRAG** | No LLM entity extraction, no community detection, no hierarchical summaries |
| **Typed precedent edges (`OVERRULES`, `DISTINGUISHES`)** | Not available in open citation data |
| **3-hop traversal** | No query class requires it |
| **LLM router** | Rules are sufficient for four classes |
| **Separate vector database** | pgvector HNSW is adequate at 2M vectors |

---

## 15. Summary

**One PostgreSQL. Four retrieval arms — dense HNSW, lexical, exact citation match, and a bounded adjacency-table citation graph. A deterministic router that sends HOLDING queries to a single RRF pass and reserves graph traversal for the three relational query classes. Per-mode and per-arm RRF constants, with the graph arm weighted low and concentrated at the top because a direct citation edge is deterministic evidence. Two-hop traversal gated behind an ablation and removed if it does not clear its Recall@10 gate. No cross-encoder reranker. A 1,500-token context cap derived from a measured TTFT curve, a static system prompt for KV-cache reuse, and a hard abstention rule.**

**Reliability is not a property of the stack. It is the outcome of clearing the seven gates in §11.3 and the filtered-recall gate in §4.3 before release.**
