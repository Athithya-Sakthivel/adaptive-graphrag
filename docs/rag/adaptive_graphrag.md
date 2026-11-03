# Adaptive GraphRAG — End-to-End Retrieval Plan

**Status:** Finalized
**Scope:** US case-law opinions, 1–2M vectors, PostgreSQL + pgvector + recursive-CTE citation graph
**Depends on:** [datasets.md](./datasets.md), [chunking.md](./chunking.md)
**Last revised:** 2026-11-03

---

## 1. What this document specifies

This is the authoritative retrieval plan. It covers the full path from user
question to cited answer:

1. query classification and routing,
2. dense, sparse, exact-match, and graph retrieval arms,
3. two-pass RRF fusion with per-mode and per-arm rank constants,
4. parent-context expansion,
5. CPU-bounded generation,
6. evaluation gates and kill criteria.

It does not cover the chunking algorithm (see `chunking.md`) or the dataset
schema (see `datasets.md`).

**Normative language.** `MUST` = a gate that blocks release. `SHOULD` = a
default that may be overridden only with a recorded ablation result.

---

## 2. Why this is "adaptive GraphRAG" — and what that claim does not include

The term is overloaded. This section fixes what it means here.

**GraphRAG** means the citation graph is a **retrieval index**, not metadata.
It is traversed during retrieval to answer questions that vector similarity
cannot: which cases cite X, and how has X's authority propagated. It is
knowledge-graph RAG over explicit `CITES` edges, not Microsoft-style GraphRAG
(no LLM entity extraction, no community detection, no hierarchical summaries).

**Adaptive** means the router selects the cheapest sufficient retrieval path per
query. It does **not** mean every query traverses the graph. Production adaptive
systems make the same bet: a cheap classifier routes the majority of easy
queries to dense retrieval and reserves expensive graph paths for the queries
that need them[reference:0].

The evidence that routing matters is direct:

| System | Routing result |
|---|---|
| **A2RAG** (ICDEW 2026) | Adaptive controller verifies evidence sufficiency and triggers targeted refinement only when needed: **+9.9/+11.8 Recall@2**, ~50% lower token cost and latency than iterative multi-hop baselines[reference:1] |
| **Keiro** (2026) | Three-tier routing: **+5.4pp context recall on complex queries, +6.1pp on multi-hop**, at 17–39% token cost across tiers[reference:2] |
| **tkarim45/rag-architectures** | "A cheap classifier in front routes the ~80% easy queries to fast dense search and reserves expensive graph/fusion paths for the queries that need them"[reference:3] |
| **SICA + GraphRAG rollout** | GraphRAG is disabled by default (`enable_graph_rag = False`), enabled per-workspace in a canary at **1 hop** first, with acceptance checks on error rate and p95 latency[reference:4] |

**The honest limitation.** Keiro reports that its **multi-hop classifier
accuracy was only 10%**, which drove an **8pp higher failure rate on simple
queries due to over-routing**[reference:5]. Routing is the entire adaptive claim,
and routing errors are the primary failure mode. §4 is therefore deliberately
rule-based and conservative, not LLM-driven.

---

## 3. Query classes

Every question is assigned exactly one class by the intent parser.

| Class | Example | Retrieval path | Graph invoked? |
|---|---|---|---|
| **HOLDING** | "What did the court hold on qualified immunity?" | Dense + sparse → RRF (one pass) | **No** |
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

**Default is HOLDING.** A query that cannot be confidently classified does not
pay the graph cost. This is the conservative bias the evidence supports: the
C→B misroute (multi-hop question sent to dense) is a structural recall-zero,
whereas the B→C misroute (simple question sent to graph) is merely expensive.
When uncertain, the parser SHOULD fall back to HOLDING and let the answer
indicate insufficient evidence rather than silently traverse the wrong graph
neighbourhood.

---

## 4. Pipeline

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

## 5. Dense retrieval

### 5.1 Index

```sql
CREATE INDEX CONCURRENTLY idx_chunks_embedding
  ON chunks USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);
```

HNSW is the baseline because it is a strong speed–recall option supported
everywhere pgvector runs. At 1–2M vectors it is comfortably within its operating
range; the failure mode is **memory**, not capacity. Real-world HNSW deployments
land at **20–25 KB per vector** once graph metadata is included, so 2M vectors
imply a working set in the tens of gigabytes[reference:6]. This is a sizing
requirement, not a reason to change engines.

### 5.2 Query with iterative scans

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

**`iterative_scan` is mandatory, not optional.** Without it, a filtered HNSW
query post-filters after the ANN scan and can return far fewer rows than
requested. The measured effect is **~50–70% filtered recall before, ~99%
after**[reference:7]. The same fix is recommended as the first step in scaling
production pgvector deployments[reference:8].

### 5.3 The filtered-recall gate

Before release, measure **Recall@50 against exact kNN** on the same filtered
subset at three selectivity levels: 1%, 10%, 50%.

| Selectivity | Required filtered recall |
|---|---|
| 1% | ≥ 95% of exact recall |
| 10% | ≥ 97% of exact recall |
| 50% | ≥ 99% of exact recall |

If the 1% case fails, add a **partial HNSW index** on the dominant filter value
(e.g. `court = 'ca9'`) or reduce filter selectivity. Do not ship a filtered
query that has not passed this gate.

---

## 6. Sparse retrieval

### 6.1 Two implementations, one role

| Implementation | When | Notes |
|---|---|---|
| **`pg_textsearch` (BM25)** | If available on the deployment | Preferred; real IDF and length normalisation |
| **`tsvector` + `ts_rank_cd`** | Always available | **Not BM25** — no IDF, no term-frequency saturation. Acceptable as a lexical ranker, not as a relevance model |

```sql
SELECT chunk_id, content,
       ts_rank_cd(tsv, websearch_to_tsquery('english', $1)) AS lex
FROM chunks
WHERE tsv @@ websearch_to_tsquery('english', $1)
ORDER BY lex DESC
LIMIT 50;
```

**Terminology discipline.** The plan MUST NOT describe `ts_rank_cd` as BM25.
Production reference implementations make this distinction explicitly:
"`ts_rank_cd` orders the matches by how close together the query terms appear.
It has no IDF weighting, unlike BM25"[reference:9].

### 6.2 What sparse retrieval is for

Sparse carries the **exact-identifier and party-name** load that dense retrieval
loses. In legal queries this is not a marginal case: citation strings, statutory
references, and docket numbers are the primary retrieval key for LOOKUP and
VALIDITY queries.

---

## 7. Exact-match retrieval

```sql
SELECT case_id, citation, title
FROM cases
WHERE citation = $1        -- e.g. '410 U.S. 113'
LIMIT 20;
```

**Exact match is a separate arm, not a filter on sparse.** BM25 tokenizes
`410 U.S. 113` into four tokens and scores it as a bag of words; a btree equality
lookup returns the exact case. The two arms are not substitutes.

---

## 8. Graph retrieval

### 8.1 Storage

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

**Referential integrity is enforced at the schema level.** Both endpoints are
foreign keys into `cases`. Dangling edges are impossible once the FK constraints
hold, which removes the silent empty-traversal failure mode.

### 8.2 Why recursive CTE and not Apache AGE

The benchmark evidence is decisive for 1–3 hop traversal:

| Depth | Recursive CTE p50 | AGE variable-length Cypher p50 |
|---|---|---|
| 1 hop | **0.34 ms** | 33.7 ms |
| 2 hops | **0.42 ms** | 52.4 ms |
| 3 hops | **0.95 ms** | 132.1 ms (p95 670 ms) |

Source: pg-raggraph engine-isolated bake-off, row-count parity ±0.2%[reference:10].

The architectural reason matters as much as the numbers: **AGE and pgvector do
not compose in one query.** The core GraphRAG operation — expand a neighbourhood,
then rank chunks by vector similarity — is a single SQL statement with recursive
CTEs. With AGE, Cypher results are wrapped in `agtype` and vector operators want
plain columns, so implementations run two round-trips[reference:11]. AGE is also
unavailable on AWS RDS, Google Cloud SQL, Supabase, and Neon[reference:12].

**The honest counter-evidence.** The same bake-off reports that in realistic
semantic→graph pipelines, AGE's one-row-per-case shape **wins the 1-hop pipeline
on latency (9.5 ms vs 37 ms p50)**, with near-parity at 2 hops[reference:13]. So
the claim is not "CTE is always faster." The claim is: for a deterministic
citation graph with chunk-granular provenance and bounded 1–2 hop traversal, CTE
is the simpler, portable, single-query option. If the graph acquires multiple
edge types and complex Cypher patterns, re-evaluate AGE.

### 8.3 1-hop (LOOKUP, VALIDITY)

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

**Cap at the query, not in application code.** A landmark case can have
thousands of incoming citations; the fan-out cap is part of the traversal
contract.

### 8.4 2-hop (APPLICATION only)

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

1. **Fan-out cap of 30 at hop 1.** Without it, expansion from a hub case is
   unbounded.
2. **`UNION` (not `UNION ALL`) for cycle safety.** Citation graphs contain
   cycles; `UNION ALL` would not terminate on them.
3. **`DISTINCT` + `MIN(hop)`.** Two-hop paths converge on hub cases; without
   deduplication the same case appears many times and dominates the fused
   ranking.

**2-hop is gated, not automatic.** It is enabled only for APPLICATION queries,
and only if the ablation in §12 shows it beats 1-hop on multi-hop Recall@10 by
≥5 points. If the ablation fails, APPLICATION falls back to 1-hop and the
two-hop claim is removed from the README.

---

## 9. Fusion — two-pass RRF

### 9.1 Why rank-based fusion

Dense cosine distance, lexical rank, and graph hop distance are on unrelated
scales. RRF fuses by **rank**, not score, which removes the need to normalise
cosine distance against `ts_rank_cd` or to invent a weight between them[reference:14].

The formula is:

```
RRF(d) = Σ_arms  weight_arm / (k_arm + rank_arm(d))
```

### 9.2 Pass 1 — seed selection (dense + sparse + exact)

```sql
WITH dense AS (
  SELECT chunk_id, ROW_NUMBER() OVER (ORDER BY dist) AS rank
  FROM ( /* dense arm, top-50 */ ) d
),
sparse AS (
  SELECT chunk_id, ROW_NUMBER() OVER (ORDER BY lex DESC) AS rank
  FROM ( /* sparse arm, top-50 */ ) s
)
SELECT chunk_id,
       SUM(CASE WHEN arm = 'dense'  THEN 1.0 / ($2 + rank)
                WHEN arm = 'sparse' THEN 1.0 / ($3 + rank)
           END) AS rrf1
FROM (
  SELECT chunk_id, rank, 'dense'  AS arm FROM dense
  UNION ALL
  SELECT chunk_id, rank, 'sparse' AS arm FROM sparse
) x
GROUP BY chunk_id
ORDER BY rrf1 DESC
LIMIT 30;
```

**Per-mode `k` for pass 1:**

| Class | Dense `k` | Sparse `k` | Rationale |
|---|---|---|---|
| HOLDING | 60 | 60 | Both arms are approximate; symmetric treatment |
| LOOKUP | 60 | 15 | Sparse is exact for citations; trust its top ranks |
| VALIDITY | 60 | 20 | Sparse resolves the case name |
| APPLICATION | 60 | 20 | Same as VALIDITY |

**`k=60` is a starting convention, not a proven optimum.** It comes from the
original RRF paper and is the most widely deployed default[reference:15], and at
least one production system re-confirmed it as its grid optimum[reference:16].
But another production tune found **k=10 won every metric** over k=20 and k=60
on its corpus[reference:17], and the literature warns that **NDCG swings wildly
as a function of RRF parameters** and that tuned values do not generalise
out-of-domain[reference:18]. Therefore:

- `k=60` is the default **because it is the published convention**, not because
  it is optimal here.
- `k ∈ {10, 30, 60, 120}` MUST be swept on this corpus before release (§12,
  experiment C).
- The graph arm's low `k` is a separate decision (§9.4), justified by the
  determinism of a direct citation edge.

### 9.3 Pass 2 — graph merge (weighted)

```sql
SELECT chunk_id,
       SUM(weight / (k + rank)) AS rrf2
FROM (
  SELECT chunk_id, rank, 0.85 AS weight, 60 AS k FROM seeds
  UNION ALL
  SELECT chunk_id, rank, 0.15 AS weight, 10 AS k FROM graph_hits
) x
GROUP BY chunk_id
ORDER BY rrf2 DESC
LIMIT 20;
```

| Arm | Weight | `k` | Rationale |
|---|---|---|---|
| Hybrid seeds | 0.85 | 60 | Already fused; rank 1 is the consensus result |
| Graph | **0.15** | **10** | A direct `CITES` edge is deterministic; low `k` concentrates its signal at the top |

**Why the graph arm gets the lowest `k`.** With `k=60`, a graph rank-1 candidate
scores `1/61 ≈ 0.016` and a graph rank-10 candidate scores `1/70 ≈ 0.014` — a
13% difference. With `k=10`, rank 1 scores `0.091` and rank 10 scores `0.050` —
an 82% difference. On a shallow, high-precision list, `k=60` flattens away the
very signal the graph arm exists to provide.

**Why the 0.85/0.15 weight split.** The graph arm supplements the hybrid
consensus; it does not replace it. This mirrors production GraphRAG defaults
that weight graph below vector and lexical fusion.

### 9.4 One pass or two — the rule

| Class | RRF passes | Arms fused |
|---|---|---|
| HOLDING | **1** | Dense + sparse (+ exact) |
| LOOKUP, VALIDITY | **1** | Sparse + exact + graph 1-hop |
| APPLICATION | **2** | Pass 1: dense + sparse → seeds. Pass 2: seeds + graph 2-hop |

**Never fuse graph and dense in the same pass.** The graph arm has a hard cut at
its fan-out cap; dense and sparse have long tails. Fusing them together lets the
hybrid tail overwhelm the graph arm's deterministic top-1.

---

## 10. Parent expansion and context budget

After pass 2, deduplicate by `chunk_id`, then expand each selected child to its
parent section.

| Parameter | Value |
|---|---|
| Expansion | child → parent (`parent_id`) |
| Context cap | **1,500 tokens** (default), 2,000 absolute maximum |
| Selection | top-N by fused score until the cap is reached |
| Re-check | re-validate `court`, `date`, `precedential_status` on the expanded parent |

**Why 1,500 and not 4,000.** On a 4-core CPU, prefill dominates latency. A
4,000-token context costs several times the time-to-first-token of a
1,500-token context. The context cap is a **latency contract**, not a quality
preference, and it must be set from a measured TTFT curve (§12, gate L2), not
from the model's maximum context length.

---

## 11. Generation

### 11.1 Server configuration

```bash
./llama-server \
  -m /workspace/models/llm/LFM2.5-1.2B-Instruct-Q4_K_M.gguf \
  -c 8192 \
  -t 4 -tb 4 \
  --jinja \
  --port 8080
```

- `-c 8192` leaves room for the context cap plus a 500-token answer.
- `-t 4 -tb 4` matches physical cores. Oversubscribing threads hurts prefill
  throughput.
- `--jinja` is required for the model's chat template.
- Warmup MUST NOT be disabled; it removes cold-start latency from the first real
  request.

### 11.2 Request contract

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

**The system prompt MUST be static.** A prompt that changes per request loses
KV-cache prefix reuse and pays full prefill on every turn. The static prompt
should be under 200 tokens and contain: role, evidence-only instruction,
per-claim citation requirement, and the abstention rule.

### 11.3 Abstention

The prompt contains a hard rule:

> If the evidence does not contain a direct answer, respond exactly
> `Insufficient evidence` and do not speculate.

**Abstention accuracy is a measured metric** (§12), not a prompt detail. A
system that answers unsupported questions is less reliable than one that refuses
them.

---

## 12. Evaluation — the release gates

### 12.1 Metric set

| Layer | Metric | Target |
|---|---|---|
| Retrieval | Recall@10, Recall@50 | Baseline-relative; no fixed absolute |
| Ranking | nDCG@10 | Baseline-relative |
| Graph | Path validity, entity-resolution precision | 100% valid paths (FK-enforced) |
| Answer | Citation precision | **≥ 90%** for decision-support workloads[reference:19] |
| Answer | Faithfulness | No more than 2% regression vs baseline[reference:20] |
| Safety | Abstention accuracy on negative queries | Measured, reported |
| Operations | p50/p95 latency, filtered recall | Measured, reported |

**Evaluation set.** 200–500 independently reviewed questions is the enterprise
pilot baseline[reference:21]. The recommended stratification is **20% simple
lookup, 40% multi-hop, 20% aggregation/comparison, 20% adversarial**[reference:22].
For this project, the deterministic citation-lookup queries in
`eval_isolated/synthesized_graph_qa/` supply the multi-hop stratum; a
hand-labelled holding-question set supplies the answer-quality stratum.

### 12.2 Ablation — four configurations

| Config | Components | Must beat |
|---|---|---|
| **A** | Dense only | — (floor) |
| **B** | Dense + sparse + exact + RRF (one pass) | A on Recall@10 and nDCG@10 by ≥3 points |
| **C** | B + graph 1-hop (per-mode k) | B on LOOKUP/VALIDITY by ≥5 points |
| **D** | C + graph 2-hop (APPLICATION only) | C on multi-hop Recall@10 by ≥5 points |

**Additional sweeps required by this plan:**

| Sweep | Values | Decision |
|---|---|---|
| **RRF k** | 10, 30, 60, 120 | Adopt the sweep winner; document it |
| **Graph k** | 5, 10, 20 | Adopt the sweep winner for the graph arm |
| **Context cap** | 1,000 / 1,500 / 2,000 tokens | Adopt the largest cap whose TTFT meets target |

### 12.3 Kill criteria

| Condition | Action |
|---|---|
| B does not beat A by ≥3 points Recall@10 | Fusion is not helping; inspect sparse coverage before proceeding |
| C does not beat B on LOOKUP/VALIDITY by ≥5 points | Graph 1-hop is decorative; drop it from those classes |
| D does not beat C on multi-hop by ≥5 points | **Drop 2-hop and remove the APPLICATION class from the README claim** |
| Filtered recall at 1% selectivity < 95% of exact | Do not ship filtered queries; add partial index or reduce selectivity |
| 1,500-token TTFT > 10 s on target hardware | Lower context cap to 1,000 tokens |
| Citation precision < 90% | Do not claim decision-support suitability |

**The kill criteria are the release contract.** A component that does not clear
its gate is removed, not defended. This is the difference between a measured
architecture and a diagram.

---

## 13. Failure modes, ranked

| Rank | Failure | Detection | Mitigation |
|---|---|---|---|
| **1** | Filtered HNSW silent recall loss | Recall@50 vs exact kNN at 1% selectivity | `iterative_scan = strict_order`; partial index if the gate fails[reference:23] |
| **2** | Router misclassification | Per-class route accuracy on the eval set | Conservative default to HOLDING; rule-based parser; log every route[reference:24] |
| **3** | Graph fan-out explosion | Candidate count after pass 1 expansion | Hard cap 30 per hop; `DISTINCT` + `MIN(hop)` |
| **4** | Prefill timeout on CPU | TTFT at 1,500 tokens | Context cap from measured curve; static system prompt for KV reuse |
| **5** | Dangling graph edges | Empty traversal from a known seed | FK constraints on `case_edges`; ingestion filter |
| **6** | Stale embeddings | Semantically wrong retrieval after content update | Cache keyed by `(chunk_id, chunker_version, embedding_model)`; re-embed on change |
| **7** | Unsupported answer | Citation precision on generated text | Abstention rule; citation validator; ≥90% precision gate |

**Failure 1 is the one that is invisible without measurement.** The system
returns plausible-looking results with 50–70% of the expected evidence missing.
No downstream component detects it, because nothing downstream knows what was
dropped.

---

## 14. Scale checkpoints

| Scale | Corpus rows | Vectors (est.) | HNSW working set (est.) | Minimum RAM |
|---|---|---|---|---|
| `min` | 5,000 | ~25K | < 1 GB | 4 GB — smoke test only |
| `medium` | 100,000 | ~500K | ~4 GB | 8 GB |
| `max` | 400,000 | ~2M | **~15–50 GB** | **32 GB minimum, 64 GB preferred** |

The HNSW working-set estimate uses the measured **20–25 KB per vector** figure
for real deployments with graph metadata, not the raw 6 KB float32 payload
[reference:25]. Sizing on the raw payload under-provisions the instance and causes
cache eviction, which degrades p95 latency sharply.

**The "~2M vectors" claim belongs to `max` only.** At `medium`, the index is
roughly 500K chunks. The README MUST state the scale at which the claim was
measured.

---

## 15. Explicit non-goals

| Not implemented | Why |
|---|---|
| **Cross-encoder reranker** | Evidence in the legal domain favours no-rerank on strong first-stage retrieval; CPU cost is disproportionate. Revisit only with a ≤30M-parameter legal reranker that clears the §12 gate |
| **Apache AGE** | 1–3 hop traversal is 1.4–47× slower than recursive CTE, cannot compose with pgvector in one query, and is unavailable on most managed PostgreSQL services[reference:26][reference:27] |
| **Microsoft-style GraphRAG** | No LLM entity extraction, no community detection, no hierarchical summaries. The graph is the citation network |
| **Typed precedent edges (`OVERRULES`, `DISTINGUISHES`)** | Not available in open citation data. VALIDITY is graph-seeded and prose-verified, not typed-edge-traversed |
| **3-hop traversal** | No query class requires it. Deeper traversal returns hub noise at unbounded cost |
| **LLM router** | Rules are sufficient for four classes; the measured failure mode of learned routers is over-routing to the expensive arm[reference:28] |
| **Separate vector database** | pgvector HNSW is adequate at 2M vectors. Revisit at 10M+ or when p95 filtered latency exceeds the latency contract |

---

## 16. Summary

**One PostgreSQL. Four retrieval arms — dense HNSW, lexical, exact citation
match, and a bounded adjacency-table citation graph. A deterministic router that
sends HOLDING queries to a single RRF pass and reserves graph traversal for the
three relational query classes. Per-mode and per-arm RRF constants, with the
graph arm weighted low and concentrated at the top because a direct citation
edge is deterministic evidence. Two-hop traversal gated behind an ablation and
removed if it does not clear its Recall@10 gate. No cross-encoder reranker. A
1,500-token context cap derived from a measured TTFT curve, a static system
prompt for KV-cache reuse, and a hard abstention rule.**

The evidence that this design is correct is that chunking and routing dominate
retrieval quality far more than engine choice: chunking accounts for 73–88% of
the gain in the strongest legal chunking study, and adaptive routing
delivers +5.4 to +11.8 Recall@2 at roughly half the cost of always-on
multi-hop[reference:30]. The evidence that it must be measured rather than
assumed is that the same adaptive routing literature reports an 8pp failure
increase from over-routing when the classifier is wrong[reference:31].

**Reliability is not a property of the stack. It is the outcome of clearing the
seven gates in §12.3 and the filtered-recall gate in §5.3 before release.**

---

## References

1. **A2RAG: Adaptive Agentic Graph Retrieval for Cost-Aware and Reliable
   Reasoning.** IEEE ICDEW 2026.
   https://www.computer.org/csdl/proceedings-article/icdew/2026/057500a187/2hmFVPwLgsg
2. **Keiro: Complexity-Aware Adaptive RAG with Three-Tier Query Routing and
   Namespace-Scoped Semantic Caching.** Zenodo, 11 June 2026.
   https://zenodo.org/records/20639902
3. **SICA + GraphRAG Rollout Runbook.** 12 February 2026.
   https://github.com/synaptent/aragora
4. **rag-architectures — Adaptive RAG (router).** 2026.
   https://github.com/tkarim45/rag-architectures
5. **pg-raggraph: PostgreSQL-native GraphRAG.** 2026.
   https://github.com/yonk-labs/pg-raggraph
6. **Why Not Apache AGE?** pg-raggraph, 2026.
   https://github.com/yonk-labs/pg-raggraph/blob/main/docs/why-not-apache-age.md
7. **StrucChunk: Structure-Aware Retrieval for Legal Document RAG.**
   arXiv:2610.26313, 19 October 2026. https://arxiv.org/abs/2610.26313
8. **Hybrid Retrieval-Augmented Generation on Serverless PostgreSQL.**
   Zenodo, 2026. https://zenodo.org/records/19394029
9. **rag-hybrid-citations — Hybrid RAG on Postgres + pgvector.** 2026.
   https://github.com/jpfiguer/rag-hybrid-citations
10. **RRF parameter optimization (ADR-0006).** ai-raccoon, 2026.
    https://github.com/Arasz/ai-raccoon
11. **Retrieval upgrades — RRF-k tune (60 → 10).** mathlas, 2026.
    https://github.com/Archerkattri/mathlas
12. **How Should Enterprises Measure GraphRAG Evaluation Metrics in 2026?**
    indexical.dev, 25 September 2026.
    https://indexical.dev/knowledge/how_should_enterprises_measure_graphrag_evaluation_metrics_in_2026.php
13. **pgvector iterative index scans for filtered queries (issue #949).**
    nexi-lab/nexus, 2025.
    https://github.com/nexi-lab/nexus/issues/949
14. **How to scale vector search in Postgres (pgvector) for RAG and AI
    agents.** ClickHouse Engineering, 19 October 2026.
    https://clickhouse.com/resources/engineering/scale-vector-search-postgres
