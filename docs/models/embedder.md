# Embedder — `dinghy-law-0.6b-v1` (OpenVINO FP16)

**Status:** Production-ready
**Artifact:** `Athithya117/dinghy-law-0.6b-v1-openvino-fp16`
**Base model:** [`Hanno-Labs/dinghy-law-0.6b-v1`](https://huggingface.co/Hanno-Labs/dinghy-law-0.6b-v1)
**Owner:** Retrieval / Ingestion

---

## 1. What this is

A 0.6B-parameter sentence-embedding model fine-tuned on legal text. It maps
queries and passages to **1024-dimensional L2-normalized vectors** so that
similar legal text lands nearby in vector space.

Shipped as an **OpenVINO IR (FP16)** so it runs on CPU without a GPU, without
cloud inference, and without needing PyTorch in the encoder's hot path.

Validated **lossless** against the PyTorch FP32 reference: query cosine 1.0,
document cosine 0.9999996, retrieval rankings bit-identical.

Not a generator. Not a reranker. Not an instruction-following model. It has
one job: text → 1024-d vector.

---

## 2. Runtime stack

| Layer | Component | Role |
| :--- | :--- | :--- |
| Orchestration | `sentence-transformers >= 6.0` | Tokenization, prompt formatting, pooling, Dense projection, L2 normalization |
| Encoder execution | `openvino >= 2026.4.1` (CPU plugin) | Runs the 0.6B-parameter transformer graph |
| IR format | OpenVINO IR (`openvino_model.xml` + `.bin`) | FP16 weights + graph topology |

Sentence-Transformers orchestrates the pipeline. OpenVINO executes the
encoder forward pass. Both are required at runtime.

**Not in the runtime stack:** ONNX Runtime, PyTorch (except the small
1024×1024 Dense head), torch's transformer weights.

---

## 3. Why FP16, not INT8 or ONNX

| Format | Disk | Accuracy | Speed on Ryzen 5700U (AVX2, no VNNI) |
| :--- | :--- | :--- | :--- |
| OpenVINO FP16 | 1136 MB | **Lossless** | = FP32 |
| OpenVINO FP32 | 2273 MB | Lossless | Baseline |
| ONNX INT8 weight-only | 1066 MB | Nominal 0.999, real discrimination gap | **8% slower than FP32** |
| ONNX FP32 | 2273 MB | Lossless | Baseline |

INT8 was exported, validated, and **rejected**: without VNNI, ONNX Runtime
dequantizes INT8 weights to FP32 before matmul, so the arithmetic is slower
than FP32 despite the smaller file. ONNX Runtime itself is not in the
production stack; it was used only during export validation.

FP16 is not a compute accelerator on AVX2-only CPUs either — the OpenVINO
CPU plugin converts FP16 constants to FP32 before computation. The gains
from FP16 are **storage and RAM** (~50% reduction), not throughput.

---

## 4. Install

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install \
  "sentence-transformers>=6.0" \
  "openvino>=2026.4.1" \
  "openvino-tokenizers>=2026.4.1" \
  "huggingface_hub>=0.30"
```

`sentence-transformers>=6.0` is required. The export uses the unified
`tokenizer.json` format; older versions that expect separate `vocab.txt` /
`merges.txt` files will fail to load the model.

---

## 5. Download

```bash
hf download Athithya117/dinghy-law-0.6b-v1-openvino-fp16 \
  --repo-type model \
  --local-dir ./models/dinghy-law-embedder
```

Or with Python:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Athithya117/dinghy-law-0.6b-v1-openvino-fp16",
    repo_type="model",
    local_dir="./models/dinghy-law-embedder",
    ignore_patterns=["*.pyc", "__pycache__/**"],
)
```

### Verify integrity

```bash
cd models/dinghy-law-embedder
sha256sum openvino/openvino_model.xml openvino/openvino_model.bin
```

Expected (from `openvino_fp16_manifest.json`):

```
3398a4ca7c562afe7efe74b7514839c92ae4c5c4efb067cb3919b9ee2eef1c33  openvino/openvino_model.xml
9e66e5c37291f7e38c3689fa9ffd9d7755016f5f1594c00c2d0655d9c2fc4771  openvino/openvino_model.bin
```

If either hash differs, re-download. Do not use a corrupted artifact.

---

## 6. Usage

### 6.1 Load once

Compile the IR once per process and reuse the instance. Reconstructing it
per request pays a ~15-second compile cost every time.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "./models/dinghy-law-embedder",
    backend="openvino",
    model_kwargs={
        "ov_config": {
            "PERFORMANCE_HINT": "LATENCY",       # "THROUGHPUT" for bulk ingestion
            "INFERENCE_NUM_THREADS": "8",        # physical core count
        }
    },
)

# Cap sequence length to your actual chunk size. Do NOT leave the default
# at the model maximum (8192); a 500-token chunk padded to 8192 wastes
# ~94% of the compute.
model.max_seq_length = 512
```

### 6.2 Encode queries and documents

The model uses an instruction prefix on queries and no prefix on documents.
Sentence-Transformers applies the correct one when you call the
corresponding method. **Do not call the generic `encode()` — it produces
document-style embeddings that silently degrade retrieval for queries.**

```python
# Query embedding (shape: (n, 1024))
query_vecs = model.encode_query(
    ["When does the statute of limitations begin for personal injury?"],
    normalize_embeddings=True,
    show_progress_bar=False,
)

# Document embeddings (shape: (n, 1024))
doc_vecs = model.encode_document(
    [
        "For a personal injury claim, the statute of limitations generally "
        "begins to run when the cause of action accrues.",
        "The discovery rule postpones accrual until the plaintiff knows or "
        "should reasonably know of the injury.",
    ],
    normalize_embeddings=True,
    show_progress_bar=False,
)

# Cosine similarity (vectors are L2-normalized, so dot product = cosine)
scores = query_vecs @ doc_vecs.T          # shape (1, 2)
```

Query prefix applied automatically by `encode_query()`:

```
Instruct: Given a web search query, retrieve relevant passages that answer the query
Query: <your query here>
```

### 6.3 Full working example

```python
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_PATH = "./models/dinghy-law-embedder"

model = SentenceTransformer(
    MODEL_PATH,
    backend="openvino",
    model_kwargs={
        "ov_config": {
            "PERFORMANCE_HINT": "LATENCY",
            "INFERENCE_NUM_THREADS": "8",
        }
    },
)
model.max_seq_length = 512

queries = [
    "When does the statute of limitations begin for a personal injury claim?",
    "What is the standard for summary judgment?",
]

documents = [
    "For a personal injury claim, the statute of limitations generally "
    "begins to run when the cause of action accrues.",
    "Summary judgment is appropriate when there is no genuine dispute as to "
    "any material fact and the moving party is entitled to judgment as a "
    "matter of law.",
    "The parol evidence rule may prevent a party from contradicting the terms "
    "of a fully integrated written agreement.",
]

q = model.encode_query(queries, normalize_embeddings=True, show_progress_bar=False)
d = model.encode_document(documents, normalize_embeddings=True, show_progress_bar=False)

assert q.shape == (2, 1024) and d.shape == (3, 1024)
assert np.allclose(np.linalg.norm(q, axis=1), 1.0, atol=1e-3)
assert np.allclose(np.linalg.norm(d, axis=1), 1.0, atol=1e-3)

scores = q @ d.T
print(scores.round(4))
# Expect a high score on the diagonal-ish pairs:
#   [[0.82  0.31  0.11]
#    [0.29  0.88  0.14]]
```

---

## 7. Performance

### 7.1 The two settings that dominate throughput

| Setting | Effect | Recommendation |
| :--- | :--- | :--- |
| `max_seq_length` | Every sequence in a batch pads to this length (or the longest in the batch) | Set to your actual chunk size (512 for 500-token chunks) |
| `batch_size` | Amortizes tokenizer and per-batch overhead | 32 for bulk ingestion; 1 for single queries |

Sorting chunks by token length before batching eliminates padding waste
within mixed-length batches.

### 7.2 Measured on reference hardware

AMD Ryzen 7 5700U (8 threads, AVX2, no VNNI), 500 chunks × 500 tokens:

| Configuration | Throughput | Time for 500 chunks |
| :--- | :--- | :--- |
| `max_len=8192, batch=2, unsorted` | ~1.2 rows/s | ~7 minutes |
| `max_len=512, batch=2, unsorted` | ~8–15 rows/s | ~35–60 seconds |
| `max_len=512, batch=32, sorted` | **~15–30 rows/s** | **~17–33 seconds** |

The 7–10× speedup comes from capping sequence length, not from FP16 or the
runtime. The additional ~2× from batching and sorting is padding
elimination, not precision.

### 7.3 Batched ingestion pattern

```python
import numpy as np
import time

def encode_sorted(model, chunks, batch_size=32):
    lengths = [len(model.tokenizer.encode(c)) for c in chunks]
    order = sorted(range(len(chunks)), key=lambda i: lengths[i])
    sorted_chunks = [chunks[i] for i in order]

    # Warm-up (pays OpenVINO compile once)
    model.encode_document(
        sorted_chunks[:batch_size],
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    start = time.perf_counter()
    vecs = model.encode_document(
        sorted_chunks,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    elapsed = time.perf_counter() - start

    # Restore original order
    restored = np.empty_like(vecs)
    restored[order] = vecs
    return restored, elapsed
```

### 7.4 When FP16 does and does not help

FP16 is not an arithmetic accelerator on Ryzen 5000 (Zen 2/3) or
Intel pre–Cascade Lake. The OpenVINO CPU plugin converts FP16 constants
to FP32 before matmul. On this class of hardware:

- **Disk and RAM:** 50% reduction, real and measurable.
- **Latency:** equal to FP32, within noise.
- **Accuracy:** lossless, validated.

On Zen 4+, Ice Lake+, Sapphire Rapids+, or GPUs with native FP16
compute, the story changes. Re-benchmark before assuming the numbers
above transfer.

---

## 8. Memory

| Component | Size |
| :--- | :--- |
| `openvino_model.bin` (FP16 weights) | 1136 MB (memory-mapped) |
| `openvino_model.xml` (graph) | 2 MB |
| `2_Dense/model.safetensors` | 4 MB |
| Tokenizer assets | ~12 MB |
| OpenVINO runtime + session | ~200–400 MB |
| **Total steady state** | **~1.4–1.6 GB** |

For reference, the FP32 equivalent is ~3 GB. On an 11 GB system running
PostgreSQL and an LLM server, plan for 2 GB of free RAM available for the
embedder.

---

## 9. Validation

### 9.1 What was validated at export

| Check | Result |
| :--- | :--- |
| Query cosine vs PyTorch FP32 | min 1.0000000 |
| Document cosine vs PyTorch FP32 | min 0.9999996 |
| Max absolute error (query) | 6.45e-06 |
| Max absolute error (document) | 1.02e-04 |
| Recall@1, Recall@5, MRR | identical |
| Top-5 Jaccard overlap | 1.000 |
| Tokenizer output hash | identical |
| FP16 fraction of IR float constants | 99.99998% |

The fixture is a synthetic smoke test (4 queries, 15 passages). It proves
the export is numerically correct. It does **not** prove the model
retrieves well on your corpus.

### 9.2 How to validate on your corpus

Before bulk ingestion, run:

1. Embed 100–200 representative chunks from your corpus.
2. Pick 10 queries with a known correct answer chunk.
3. Confirm the correct chunk appears in the top-5 for each query.
4. Confirm that unrelated chunks score meaningfully lower.

If any of those fail, the problem is chunk size, prompt formatting, or
metadata filtering — not the embedder.

---

## 10. Storage

The model outputs L2-normalized `float32` vectors. With normalization on,
**cosine similarity and inner product are equivalent**, so either HNSW
`vector_cosine_ops` or `vector_ip_ops` works in pgvector.

```sql
CREATE TABLE chunk_embeddings (
    chunk_id      BIGINT PRIMARY KEY REFERENCES chunks(chunk_id),
    embedding     VECTOR(1024) NOT NULL,
    model_version TEXT NOT NULL DEFAULT 'dinghy-law-0.6b-v1-openvino-fp16'
);
CREATE INDEX ON chunk_embeddings
    USING hnsw (embedding vector_cosine_ops);
```

The `model_version` column is not decorative. When you change embedder
revision, every vector in the index becomes invalid. Re-embed the entire
corpus into a fresh table or collection; do not mix old and new vectors
in the same index.

---

## 11. Troubleshooting

| Symptom | Cause | Fix |
| :--- | :--- | :--- |
| `tokenizer.json` fails to load | `sentence-transformers < 6.0` | Upgrade to 6.0+ |
| `OpenVINO CPU plugin unavailable` | Missing `openvino` wheel | `pip install openvino`; check `ov.Core().available_devices` |
| First call takes ~15 seconds, later calls fast | Compile cost is paid once | Cache the `SentenceTransformer` instance |
| Retrieval quality collapses | `encode()` used instead of `encode_query()` | Use the query/document methods |
| Throughput ~1 row/s on 500-token chunks | `max_seq_length` left at 8192 | Set to 512 |
| `RuntimeError: unknown graph input` | Loaded IR directly with `ov.Core().read_model()` | Load via `SentenceTransformer(..., backend="openvino")` |

---

## 12. Reference

- **Model card:** https://huggingface.co/Athithya117/dinghy-law-0.6b-v1-openvino-fp16
- **Base model:** https://huggingface.co/Hanno-Labs/dinghy-law-0.6b-v1
- **Validation report:** `openvino_fp16_validation.json` (in the repository)
- **Provenance manifest:** `openvino_fp16_manifest.json` (in the repository)
- **OpenVINO CPU plugin:** https://docs.openvino.ai/2026/openvino_docs_OV_UG_supported_plugins_CPU.html
- **Sentence-Transformers OpenVINO backend:** https://sbert.net/docs/sentence_transformer/usage/efficiency.html

---

## 13. Versioning

| Field | Value |
| :--- | :--- |
| Base model revision | `080ba18e73c85ce8a513241b011b867174883175` |
| Distributed artifact | tagged release on HF repository |
| Embedding dimension | 1024 (fixed) |
| Max sequence length | 8192 (design); 512 recommended |
| Weight precision | FP16 |
| Quantization | none |
| XML SHA-256 | `3398a4ca7c562afe7efe74b7514839c92ae4c5c4efb067cb3919b9ee2eef1c33` |
| BIN SHA-256 | `9e66e5c37291f7e38c3689fa9ffd9d7755016f5f1594c00c2d0655d9c2fc4771` |

### When to re-embed the corpus

Re-embedding invalidates every stored vector and every retrieval metric.
Trigger it only when:

1. The embedding model revision changes.
2. The embedding dimension changes.
3. The pooling strategy changes.
4. The chunking strategy changes such that chunk boundaries shift.
5. A validated replacement embedder shows a meaningful Recall@10 improvement
   on the legal evaluation fixture.

Changing `max_seq_length`, `batch_size`, or `PERFORMANCE_HINT` affects
throughput only and does **not** require re-embedding.
