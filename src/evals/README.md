# Evaluation Framework

Corpus-guarded evaluation harness for the Adaptive GraphRAG retriever. Builds deterministic gold sets from the live database, executes retrieval through the production adapter, and reports metrics with bootstrap confidence intervals and per-query failure mode categorization.

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ custom_eval_    │────▶│   run_eval.py    │────▶│  results/*.json │
│ builder.py      │     │   (orchestrator) │     │  (JSON reports) │
└─────────────────┘     └──────────────────┘     └─────────────────┘
        │                        │
        │                        │
        ▼                        ▼
┌─────────────────┐     ┌──────────────────┐
│ source.py       │     │ adapter.py       │
│ (DB candidate   │     │ (EvalRuntime     │
│  loader)        │     │  protocol)       │
└─────────────────┘     └──────────────────┘
```

**Flow:**
1. `custom_eval_builder.py` queries the live database via `source.py`, stratifies candidates by court/year/indegree, and writes a JSONL gold set with embedded corpus fingerprint.
2. `run_eval.py` loads the gold set, validates the corpus fingerprint against the live database, executes retrieval via `adapter.py`, computes metrics, and writes a JSON report.
3. `test_e2e_locally.sh` orchestrates the pipeline, runs RRF k sweeps, and formats summary tables.

## File Structure

```
src/evals/
├── __init__.py              # Lazy imports
├── adapter.py               # EvalRuntime protocol + retriever adapter
├── custom_eval_builder.py   # Gold set generator (stratified sampling)
├── fingerprint.py           # SHA-256 corpus state hashing
├── metrics.py               # Recall, NDCG, MRR, bootstrap CIs, failure modes
├── run_eval.py              # Eval orchestrator with corpus drift detection
├── source.py                # Database candidate loader
├── test_e2e_locally.sh      # Bash harness for local testing
└── gold/                    # Generated gold sets (gitignored)
    ├── custom_smoke.jsonl
    └── custom_full.jsonl
```

## Gold Set Generation

### Stratification

Gold queries are sampled deterministically from the live corpus using proportional stratification across three dimensions:

| Dimension | Buckets |
|-----------|---------|
| Court | Normalized court slug (e.g., `ca9`, `scotus`) |
| Year | 5-year buckets (e.g., `1980-1984`, `2000-2004`) |
| Citation indegree | `0`, `1`, `2-4`, `5+` |

Allocation is proportional to population size within each stratum. Shuffling within strata uses a SHA-256-derived seed for reproducibility.

### Query Templates

| Class | Template | Gold Evidence |
|-------|----------|---------------|
| HOLDING | First 400 chars of holding chunk text (case name removed) | `[case_id]`, `[holding_chunk_id]` |
| LOOKUP | `"Which cases cite {case_name}?"` | `[citer_case_ids]` |
| VALIDITY | `"Has {case_name} been overruled?"` | `[overruling_citer_ids]` |
| negative_control | `"Which cases cite {case_name}?"` (indegree=0) | `[]` |

### Eligibility Criteria

- **HOLDING:** Case has a detected holding chunk (`section = 'holding'`).
- **LOOKUP:** Case has `indegree >= 1` and at least one citer in the corpus.
- **VALIDITY:** Case has `indegree >= 1` and at least one citer whose parent chunk contains overruling language (`overrul|revers|vacat` stems).
- **negative_control:** Case has `indegree = 0`.

### Corpus Fingerprint

Each gold row embeds a SHA-256 digest of the corpus state at generation time:

```python
fingerprint = SHA256(
    schema_version ||
    chunker_version counts ||
    sorted chunk_ids ||
    sorted case_ids ||
    sorted (source_id, target_id, relation) edges
)
```

The eval runner refuses to score if the live corpus fingerprint diverges from the gold set's fingerprint. This prevents silent chunker drift — measuring retrieval quality against a gold set whose chunk boundaries no longer match the index.

## Evaluation Runner

### Corpus Drift Detection

Before executing retrieval, `run_eval.py` computes the live corpus fingerprint and compares it to the gold set's embedded hash. On mismatch:

```
RuntimeError: CORPUS HASH MISMATCH. Gold=48dc0f5b...; Actual=f9dfa64b...
```

This is a hard failure. Rebuild the gold set with `--rebuild-gold`.

### Metrics

All metrics are computed per-query, then aggregated with bootstrap confidence intervals (n=1000, 95% CI).

| Metric | Definition |
|--------|------------|
| `case_recall@k` | Fraction of gold case IDs present in top-k retrieved case IDs |
| `case_ndcg@k` | Normalized discounted cumulative gain at rank k |
| `case_mrr` | Mean reciprocal rank (1/rank of first gold case ID) |
| `empty_result_rate` | Fraction of queries returning zero hits |
| `refusal_accuracy` | Fraction of negative controls returning zero hits |

### Failure Mode Taxonomy

Every query is categorized into exactly one failure mode:

| Mode | Condition |
|------|-----------|
| `success` | Gold case IDs present in top-k (or negative control returns empty) |
| `classification` | Router assigned wrong query class |
| `resolution` | LOOKUP/VALIDITY target case could not be resolved (no `graph_1hop` in arms_used) |
| `retrieval` | Gold case IDs absent from all retrieved results |
| `ranking` | Gold case IDs present in results but outside top-k |
| `empty` | Answerable query returned zero hits |
| `false_positive` | Negative control returned non-empty results |

Failure modes are reported per-class in the JSON report:

```json
"failure_modes": {
  "HOLDING": {"success": 35, "classification": 15},
  "LOOKUP": {"success": 48, "resolution": 2}
}
```

## Adapter Protocol

The eval framework communicates with the retriever through the `EvalRuntime` protocol:

```python
class EvalRuntime(Protocol):
    def corpus_fingerprint(self) -> str: ...
    def retrieve(self, query: str, *, top_k: int) -> dict: ...
    def close(self) -> None: ...
```

`adapter.py` implements this protocol by wrapping the production `retriever.route()` function. The adapter supports:

- **Arm ablation:** Disable specific retrieval arms (e.g., `--no-graph`) to measure individual arm contributions.
- **RRF k sweeps:** Override all `k` values via `--rrf-k` to tune fusion parameters.
- **Exact baseline:** Disable HNSW and force sequential scan to measure recall ceiling.
- **Metadata filters:** Pass `Filters(published_only=True)` to evaluate filtered retrieval.

## Usage

### Build Gold Set

```bash
# Smoke tier (8 queries: 2 per class)
python -m evals.custom_eval_builder \
  --source evals.source:build_source \
  --tier smoke \
  --out src/evals/gold/custom_smoke.jsonl

# Full tier (150 queries: 50 HOLDING, 50 LOOKUP, 30 VALIDITY, 20 negative)
python -m evals.custom_eval_builder \
  --source evals.source:build_source \
  --tier full \
  --out src/evals/gold/custom_full.jsonl
```

### Run Evaluation

```bash
# Baseline evaluation
python -m evals.run_eval \
  --gold src/evals/gold/custom_full.jsonl \
  --adapter evals.adapter:build_runtime \
  --suite custom \
  --top-k 10 \
  --out results/eval_full.json \
  --allow-unverified

# With metadata filter
python -m evals.run_eval \
  --gold src/evals/gold/custom_full.jsonl \
  --adapter evals.adapter:build_runtime \
  --suite custom \
  --top-k 10 \
  --published-only \
  --out results/eval_full_published.json \
  --allow-unverified

# Exact-scan baseline (bypass HNSW)
python -m evals.run_eval \
  --gold src/evals/gold/custom_full.jsonl \
  --adapter evals.adapter:build_runtime \
  --suite custom \
  --top-k 10 \
  --exact-baseline \
  --out results/eval_exact_baseline.json \
  --allow-unverified
```

### Local Harness

```bash
# Standard eval
bash src/evals/test_e2e_locally.sh --tier full

# RRF k sweep
bash src/evals/test_e2e_locally.sh --tier full --find-optimal-rrf-k

# With metadata filter
bash src/evals/test_e2e_locally.sh --tier full --published-only

# Rebuild gold set
bash src/evals/test_e2e_locally.sh --tier full --rebuild-gold
```

## Output Format

### JSON Report

```json
{
  "schema_version": 1,
  "suite": "custom",
  "config": "e2e_full",
  "top_k": 10,
  "n_queries": 150,
  "corpus_hash": "f9dfa64b09a8f6df...",
  "per_class": {
    "HOLDING": {
      "n": 50,
      "n_answerable": 50,
      "n_case_scored": 50,
      "case_recall@5": 0.880,
      "case_recall@5_ci": [0.800, 0.960],
      "case_recall@10": 0.880,
      "case_recall@10_ci": [0.800, 0.960],
      "case_ndcg@10": 0.865,
      "case_mrr": 0.860,
      "empty_result_rate": 0.02,
      "refusal_accuracy": 0.0,
      "latency": {"p50_ms": 137.3, "p95_ms": 261.1, "mean_ms": 149.1}
    },
    "LOOKUP": { ... },
    "VALIDITY": { ... },
    "negative_control": { ... }
  },
  "failure_modes": {
    "HOLDING": {"success": 35, "classification": 15},
    "LOOKUP": {"success": 48, "resolution": 2},
    "VALIDITY": {"success": 28, "retrieval": 2},
    "negative_control": {"false_positive": 18, "success": 2}
  },
  "macro_positive": {
    "case_recall@10": 0.929
  },
  "latency": {"p50_ms": 68.0, "p95_ms": 237.0, "mean_ms": 102.0}
}
```

### Summary Table

```
==============================================================================
EVALUATION SUMMARY  (top_k=10, n_queries=150)
==============================================================================
  class            N  recall@5               recall@10              ndcg@10  mrr
  -------------- ---  ---------------------- ---------------------- -------- --------
  HOLDING         50  0.880 [0.800, 0.960]   0.880 [0.800, 0.960]   0.865    0.860
  LOOKUP          50  0.953 [0.892, 1.000]   0.958 [0.897, 1.000]   0.910    0.890
  VALIDITY        30  0.907 [0.814, 0.983]   0.950 [0.867, 1.000]   0.929    0.938
  negative_control 20  —                      —                      —        —

  Failure modes (failed queries only):
    HOLDING            classification=18, retrieval=1
    LOOKUP             resolution=1, retrieval=1
    VALIDITY           retrieval=1
    negative_control   false_positive=18

  Latency: p50=68ms  p95=237ms  mean=102ms
  Macro recall@10 (positive classes): 0.929
==============================================================================
```

## Known Limitations

1. **HOLDING pool size:** Only cases with detected holding chunks are eligible. In a 9,536-case corpus, this yields ~123 eligible cases (1.3%). The 50-query HOLDING eval is a near-census of this narrow subset.

2. **Overruling heuristic:** VALIDITY gold uses substring matching on stems `overrul|revers|vacat`. This produces false positives (e.g., "lower court reversed") and false negatives (e.g., "abrogated"). Manual spot-checking recommended before claiming verified.

3. **Negative control refusal:** The retrieval layer has no refusal mechanism. Negative controls (indegree=0 cases) return BM25 matches on the case's own text, yielding `refusal_accuracy = 0.0`. Refusal belongs in the generation layer, not retrieval.

4. **Unverified gold:** All generated gold rows have `verified=false`. Human review is required before claiming verified status. The eval runner refuses to score unverified gold unless `--allow-unverified` is passed.
