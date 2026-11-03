# Decision: No Reranker

**Decision:** Ship hybrid retrieval (dense + sparse + RRF + graph) without a cross-encoder reranker.

**Why:**
- LegalBench-RAG found the best configuration was RTCS **without** a reranker; the general-purpose Cohere reranker was inferior on legal text ([arXiv:2408.10343](https://arxiv.org/abs/2408.10343)).
- On LegalBench-RAG privacy_qa, MiniLM cross-encoders applied on every agent retrieval call cut citation-span overlap from **0.825 to 0.557–0.608** vs no reranker ([OpenContracts #1378](https://github.com/Open-Source-Legal/OpenContracts/issues/1378)).
- Universal reranking degrades performance when first-stage retrieval is already strong; gated reranking improved Recall@1 ~6% and Recall@3 ~10% ([When to Rerank, ICEIS 2026](https://doi.org/10.5220/0014803000004018)).
- On 4 CPU threads, BGE-reranker-v2-m3 took **3.04 s** for 20 passages vs 12.6 ms for embedding ([measured](https://jacar.es/en/local-reranking-llama-cpp-pgvector/)); FlashRank is 5–15 ms but unproven on legal retrieval ([issue](https://github.com/stratalab/strata-core/issues/1636)).

**What we ship instead:** dense HNSW + BM25/tsvector + RRF fusion + recursive-CTE citation graph + metadata filters.

**Revisit if:** a legal-native reranker (≤30M params) improves citation precision on hard, multi-hop queries in our own ablation.
