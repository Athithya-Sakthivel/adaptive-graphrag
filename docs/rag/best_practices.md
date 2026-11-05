# High-quality retrieval at scale — what the 2026 evidence actually says

Synthesized from the current literature, ranked by **measured effect size** on legal retrieval. The headline: chunking and graph traversal dominate; embedder choice and fusion-parameter micro-tuning matter far less than most teams assume.

---

## 1. The effect-size hierarchy (this is the whole answer in one table)

| Lever | Measured effect | Source |
|---|---|---|
| **Structure-aligned chunking** | **+104% Char-Recall@8**; accounts for **73–88% of total gain** | StrucChunk |
| **Citation-graph traversal added to vector** | **40% → 70% recall (+75%)** on 500K U.S. cases | Azure HorizonDB benchmark |
| **Hybrid dense+sparse + RRF** | +15–30% recall vs single-method; BEIR nDCG@10 43.4→52.6 | Qdrant tuning guide |
| **Metadata injection into chunk text** | Improves precision+recall on proviso/condition queries; reduces hallucination without larger context | Smart Indexing, IEEE |
| **Learned cascade routing** | 74–80% EM at 20–50% graph invocation vs 70–75% at 50–80% | Learned vs Heuristic Thresholds |
| **halfvec / FP16** | **−0.5% mean recall, −49% storage**, +30–35% index speed | pinecone benchmark, lore-weave |
| **RRF k tuning** | nDCG@10 swings 0.695→0.716 between k=60 and k=10 | T2-RAGBench via Qdrant |
| **Embedder swap** | **2.71pp** — a ninth of the chunking gain | StrucChunk confound analysis |

**Read that last row twice.** The chunking architecture gain (+24.59pp) exceeds the embedder gap between OpenAI text-embedding-3-large and BGE-M3 (2.71pp) **by a factor of nine**. Every hour spent choosing an embedding model instead of fixing chunking is misallocated.

---

## 2. Chunking — the dominant lever, and it's not close

### 2.1 Structure alignment beats everything

StrucChunk, evaluated on five corpora, three legal traditions, two languages:

| Metric | Recursive splitting | StrucChunk |
|---|---|---|
| Boundary precision (CrPC) | 7.7% | **100%** |
| Cross-reference preservation (CrPC) | 0% (0/42) | **97.6%** (41/42) |
| Section coverage (GDPR) | 0% | **100%** |
| Chunks needed for 100% coverage (GDPR) | — | **−40.3% fewer** |
| Char-Recall@8 (LegalBench-RAG, 6,858 pairs) | 23.59% | **48.18%** |
| Recall@5 (CrPC, 501 queries) | 0.557 | **0.804** |
| HR@5 (Thai NitiBench-CCL, 3,730 queries) | 0.594 | **0.856** |



The German Legal Code study independently confirms the direction and adds a cost dimension: **section- and subsection-aligned chunking achieves the highest recall, while RAPTOR, Lumber, semantic clustering, and LLM-contextual chunking all perform worse — and cost far more to build.**

**What this means for your repo**: `chunker.py`'s structure-aware approach is correct. The remaining work is not architectural, it's the boundary-artifact class you already isolated (page markers, OCR corruption). Do not swap it for semantic chunking.

### 2.2 Chunks must be semantically self-contained

Smart Indexing (IEEE, Apr 2026) embeds **Act, Chapter, Section ID, and section title directly into each fine-grained chunk**, then uses parent-child retrieval — small metadata-rich clauses for precision, expanded to parent sections for generation. Result: better precision and recall on proviso/condition-sensitive queries, **lower hallucination, no larger context window, no model complexity increase**.

Your breadcrumb `[case, citation (year) · court · opinion_type · § section_path]` is exactly this pattern. The evidence validates it — and implies you should **not** move metadata to side-columns only.

### 2.3 Chunk size: structure beats fixed windows, but "contextual ~200 words" is the practical sweet spot

The Thai Royal Gazette study (19 configurations across chunk sizes, boundary strategies, contextual chunks, atomic facts, whole documents) found: **contextual 200-word chunks with hybrid retrieval give the strongest practical balance of evidence coverage, answer quality, and query cost.**

Your child target is 320–400 tokens (~240–300 words) — slightly larger than that finding, but the two aren't directly comparable (words vs tokens; case law vs gazette). Worth a sweep, not a rewrite.

---

## 3. Hybrid retrieval + RRF — necessary, but the parameters are where quality lives

### 3.1 Hybrid is the robust default across legal corpora

- Brazilian JUÁ benchmark (4 heterogeneous legal scenarios): **RRF-based systems achieve the best aggregate effectiveness**, with statistically significant gains over BM25 and most dense models; a domain-tuned 4B model leads nDCG@10 and MRR@10, an 8B leads MAP@10 and Recall@10.
- SCOTUS case law: BM25Okapi + BGE-M3 weighted hybrid → **Recall@5 = 0.7812, MRR@5 = 0.7720**, Context Precision 0.8052.
- Caveat worth internalizing: on one of five Qdrant datasets (DBPedia-entity), **dense alone beat the fused result** (0.4677 vs 0.4638). Hybrid is worth measuring, not assuming.

### 3.2 RRF `k` is not a detail — sweep it per-arm

The evidence contradicts the "k=60 is the answer" folklore:

- Original RRF paper: k=60. Qdrant ships **k=2**. Elasticsearch defaults `rank_constant` 60.
- **T2-RAGBench 2026: RRF k=10 → 0.716 Recall@5; k=60 → 0.695.**
- **Per-arm asymmetry beats symmetric k**: "performance improves off-diagonal, where the parameter takes on different values for the semantic and lexical components." On MS MARCO NDCG improves when η_LEX > η_SEM; on HotpotQA (out-of-domain) the opposite holds.
- **But tuned parameters do not generalize out-of-domain.**

**What this means for your repo**: `fusion.py` already takes `k_per_arm` as a parameter — that's the right design. Sweep `{10, 30, 60, 120}` per arm on your own eval set and **document the sweep result**, because the literature says the optimum is corpus-specific and non-transferable. Do not hardcode 60 and call it principled.

### 3.3 The two-stage pattern: RRF candidate pool → fine-grained comparison

Section-Weighted Hybrid (IEEE, 2026) is the strongest architectural template for legal case retrieval:

- **Stage 1**: parallel BM25 + dense whole-document search fused by RRF → high-recall candidate pool.
- **Stage 2**: fine-grained **like-for-like** comparisons (query *reasoning* vs candidate *reasoning*, query *facts* vs candidate *facts*), after segmenting judgments into facts / issues / decision / reasoning.
- **Critical detail**: apply **query-wise Z-score normalization** before aggregating, because BM25 scores are unbounded while cosine similarity is bounded.

This is a better use of a second stage than a generic cross-encoder reranker, because it compares **legally corresponding facets**, not surface text.

---

## 4. pgvector at scale — the three things that silently break

### 4.1 Filtered recall collapse is THE production failure mode

The mechanism, precisely: pgvector's HNSW applies the filter **after** the ANN scan, bounded by `ef_search`. As `k` grows, you must extract more valid results from a fixed-size candidate set, so **recall decreases systematically with k** — unless you proportionally scale `ef_search`. At low selectivity, a majority of the raw-vector candidates fail the predicate and true neighbours are never examined.

Mitigations, in order of importance:

1. **pgvector ≥0.8 iterative index scans** — the index keeps scanning until enough rows pass the `WHERE`. This is the single highest-value configuration change.
2. **Scale `ef_search` with `k`** — the literature is explicit that holding `ef_search` fixed while increasing `k` guarantees recall decay.
3. **Know the crossover**: under **low selectivity**, IVFFlat can beat HNSW because cluster-pruning adapts better to sparse regions of vector space; pgvector's post-filtering beats FAISS pre-filtering for **high-selectivity** filters.

**Your `dense.py` must `SET LOCAL hnsw.iterative_scan` inside the transaction** — and you should add a filtered-recall gate (Recall@50 vs exact kNN at 1%/10%/50% selectivity) to `eval/`, because this is the one failure that produces plausible-looking results with 30–50% of evidence missing.

### 4.2 HNSW memory: 20–25 KB/vector, not 6 KB

Raw FP32 at 1,536 dims is ~6 KB, but **pointers, neighbour lists, and graph metadata push real deployments to 20–25 KB/vector**. At 1M vectors that's tens of GB; at 10M, memory pressure becomes the central design concern.

Your 1024-dim `halfvec` at ~2M chunks lands around **20–40 GB working set** — consistent with the earlier estimate in your own docs. This is a **sizing requirement**, not a reason to change engines.

### 4.3 halfvec is a real trade, and the trade is excellent

Two independent 2026 benchmarks:

| Finding | Value |
|---|---|
| Recall cost vs FP32 (wiki_dpr, 1M×768d) | **≤0.0002 difference across 531 sweep points** |
| Recall cost (real passage corpus) | **−0.5% mean, worst query 0.90** |
| Storage saved | **49%** (index 4.1 GB → 2.0 GB) |
| Speed at equal recall | **30–35% faster** (ef 80: 1.75 ms vs 2.50 ms) |
| FP32 rerank over FP16 | **Buys nothing**, adds 0.2–0.4 ms |



Your `halfvec(1024)` choice is validated. One caveat from the same benchmark: **pgvectorscale 0.9.0 has no halfvec opclass for DiskANN** — so "StreamingDiskANN vs halfvec" is not a comparison anyone can run. If you ever consider DiskANN, that's a blocker.

---

## 5. Citation graph — your differentiator, with a hard number attached

The Azure benchmark on the **U.S. Case Law dataset (500K cases)** is the most directly relevant number available:

> **Vector search alone: 40% recall. Adding graph-augmented retrieval with citation traversal: 70% recall — a 75% improvement.**

The stated reason is exactly the structural-insight argument: *"the correct result is structurally important but semantically distant — a legal precedent cited by 200 cases ranks low on embedding similarity but high on citation authority."*

Related confirmations:
- Dual-layer Legal-RAG (10K Indian SC opinions + CrPC): A-RAG + graph retrieval moved accuracy from **63% → 77%**, statistically significant.
- Traditional RAG achieves only **32–75% accuracy on multi-hop tasks** — the gap graph traversal closes.
- Personalized PageRank over the citation KG is used for cross-lingual Swiss legal citation retrieval.

**Two operational warnings**:
1. **Latency**: multi-turn graph tool chains take **20–35 seconds** per query in the Neo4j blueprint. Your recursive-CTE approach avoids this by staying in one SQL statement — a genuine advantage worth preserving and documenting.
2. **Memory-bounded projection**: production legal GraphRAG systems use **pressure-capped, durable graph projection** to avoid unbounded memory during citation passes.

---

## 6. Routing — learned beats fixed thresholds, but only where the features exist

The controlled comparison (8 Indian Bare Acts, 200 QA pairs, Fact Lookup / Citation Verification / Precedent Chaining):

| Strategy | Exact Match | Graph invocation rate |
|---|---|---|
| Fixed-threshold cascade | 70–75% | 50–80% |
| **Learned-threshold cascade** | **74–80%** | **20–50%** |

The learned router is a **lightweight logistic regression on four features**: top-1 vector similarity, mean top-k similarity, number of distinct source documents in results, normalised query length. Trained on 50 labelled examples.

**The critical caveat**: the advantage is **strongest for Fact Lookup** and **limited for Citation Verification and Precedent Chaining** — "more complex multi-hop legal reasoning requires stronger graph reasoning and retrieval methods."

**What this means for your repo**: your deterministic rule-based router is defensible for now, and it's the correct starting point (the docs cite Keiro's 10% multi-hop classifier accuracy as the reason). But the evidence says a **4-feature logistic regression** is a low-cost upgrade path once you have ~50 labelled examples — and it directly addresses your `gate_8`-style risk of over-routing. Keep the rules as the cold-start default; add the learned threshold as a measured upgrade.

---

## 7. Evaluation protocol — how to report so the numbers mean something

### 7.1 Metrics set (this is the convention the field is converging on)

| Metric | Use |
|---|---|
| **Recall@k** (k ∈ {5, 10, 20, 50}) | Coverage — the primary retrieval metric |
| **MRR@k** | Position of first relevant result |
| **nDCG@10** | Graded ranking quality at the top |
| **MAP@10** | Especially important for cross-reference queries with multiple expected sections |
| **Char-level P@k / R@k** | The official LegalBench-RAG protocol — not numerically comparable to section-level metrics |
| **Hit@k** | Binary coverage, common in compliance corpora |



### 7.2 Statistics — do not report point estimates alone

StrucChunk's protocol: **Wilcoxon Signed-Rank Test (two-sided, paired per query) with Benjamini–Hochberg correction, Cohen's d effect sizes, 95% bootstrap confidence intervals on Recall@5 differences.**

### 7.3 Realistic baseline numbers to calibrate against

| System / corpus | Numbers |
|---|---|
| SCOTUS hybrid sparse-dense | R@5 0.7812, MRR@5 0.7720, Context Precision 0.8052 |
| Hybrid lexical, EU AI Act/GDPR/DORA compliance | Hit@1 0.360, Hit@10 0.770, R@10 0.392, MRR 0.493, nDCG@10 0.344 |
| Legal-HybridRAG (Chinese) | R@10 43.0%, MRR 0.3290 |
| Hybrid lexical, Lex9165 | P@20 5.5%, R@20 79.7%, Hit@20 94%, MAP@20 64.4%, MRR@20 77.4%, nDCG@20 71.3% |

**Note the spread.** R@10 of 0.392 and 0.43 on compliance corpora are normal — not a sign of a broken system. High Hit@k with low Recall@k means you find *something* relevant but not the *complete* obligation set, which is the characteristic legal-retrieval failure.

### 7.4 The domain-shift warning you must not ignore

Legal-HybridRAG achieved **43.0% R@10 on its Chinese dataset** and **4.38% R@10 on external LeCaRD validation** — described as "distributional sensitivity."

**Implication**: a single-corpus eval is not evidence of generalization. Your portfolio claim must state the corpus, and any cross-corpus claim needs a second benchmark (LegalBench-RAG is the natural choice, since StrucChunk's 6,858 query-passage pairs are public and char-level protocol is defined).

---

## 8. What the evidence says to *not* do

| Practice | Why the 2026 evidence rejects it |
|---|---|
| **Semantic / RAPTOR / Lumber chunking** | Section-aligned chunking beats them on recall *and* is far cheaper; "simpler methods also offer favorable computational efficiency compared to LLM-intensive techniques." |
| **Fixed-size chunking** | 2.8–7.7% boundary precision vs 100%. |
| **Embedder-first optimization** | 2.71pp vs 24.59pp — 9× smaller lever. |
| **RRF k=60 uncritically** | T2-RAGBench k=10 beat k=60 (0.716 vs 0.695); Qdrant ships k=2. |
| **Symmetric RRF k across arms** | Off-diagonal (asymmetric) consistently better. |
| **Hand-tuned fusion weights without measurement** | Qdrant docs: "unlikely to beat the default." |
| **Holding `ef_search` fixed while raising `k`** | Guarantees recall decay under filtering. |
| **FP32 rerank over FP16 index** | "Buys nothing here and costs latency." |
| **Fixed-threshold cascade routing** | Learned router reaches higher EM with 30–60% fewer graph invocations. |
| **Skipping statistical significance** | The field convention now includes Wilcoxon + BH + bootstrap CIs. |

---

## 9. Concrete action list mapped to your repo

| Priority | Action | File | Expected effect |
|---|---|---|---|
| **P0** | Add filtered-recall gate: Recall@50 vs exact kNN at 1%/10%/50% selectivity | `eval/filtered_recall.py` | Catches the #1 silent failure before scale |
| **P0** | Verify `SET LOCAL hnsw.iterative_scan` is inside every dense query transaction | `retriever/dense.py` | Fixes the 50–70% filtered recall loss |
| **P1** | Sweep RRF `k` per arm over `{10, 30, 60, 120}`; record winner + per-arm values | `eval/rrf_sweep.py` | 0.695→0.716-class gains, corpus-specific |
| **P1** | Build the 4-config ablation (dense / +sparse+RRF / +graph-1hop / +graph-2hop) | `eval/rrf_sweep.py` | Quantifies the graph's +75% claim *on your corpus* |
| **P1** | Add graph traversal to a measured baseline and report the delta | `retriever/graph.py` + eval | The differentiator; needs a number attached |
| **P2** | Z-score normalise before any score-level (non-rank) fusion | `retriever/fusion.py` | Required if you ever move beyond pure RRF |
| **P2** | Two-stage: RRF candidate pool → facet-aligned comparison | future | Stronger than generic reranking on legal text |
| **P2** | Measure `halfvec` recall delta against a FP32 control on your corpus | `eval/` | Confirms the −0.5% / −49% storage trade locally |
| **P3** | Collect 50 labelled queries; train 4-feature logistic router | `retriever/router.py` | 74–80% EM at 20–50% graph invocation |
| **P3** | Report Wilcoxon + BH + bootstrap CIs alongside every metric | `eval/` | Field-standard rigour; distinguishes portfolio from toy |

---

## 10. The one-paragraph thesis

**Chunking architecture and citation-graph traversal are the two levers that matter, and both are already in your design.** The 2026 evidence puts structure-aligned chunking at +104% Char-Recall@8 and 73–88% of total retrieval gain, with the embedder accounting for a ninth of that; and it puts citation-graph augmentation at 40%→70% recall on 500K U.S. cases. Everything else — halfvec, RRF parameter sweeps, learned routing, Z-score normalization, two-stage facet comparison — is a 0.5–5 point refinement on top. The failure modes that will actually hurt you at scale are **filtered HNSW recall collapse** (silent, 30–50% evidence loss) and **domain shift** (43%→4.4% R@10 on a foreign corpus). Both are caught by evaluation gates, not by architecture changes.
