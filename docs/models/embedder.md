# Embedder — `bge-small-en-v1.5-openvino-qint8`

**Status:** Production
**Artifact:** `Athithya117/bge-small-en-v1.5-openvino-qint8` @ `2d7762ea1c5aa69975d766c7b93c15e82eb645ce`
**Base model:** `BAAI/bge-small-en-v1.5`
**Served by:** `src/dense-service` (FastAPI + OpenVINO)

---

## What it is

384-dim L2-normalized text embeddings for legal RAG. Ships as an **int8 OpenVINO IR**, baked into the dense-service image at build time. No PyTorch, no FastEmbed, no ONNX Runtime, no Hub calls at runtime.

Indexing pipeline (`indexing_pipeline`) and the future retriever are both HTTP clients of the same `POST /embed` endpoint — that is what keeps query and document vectors in one space.

---

## Runtime stack

| Layer | Component |
|---|---|
| HTTP | FastAPI + uvicorn (`--workers 1`) |
| Encoder | `openvino` CPU plugin, static shape `(DENSE_BATCH_SIZE, DENSE_MAX_SEQ_LENGTH)` |
| Tokenizer | `transformers.AutoTokenizer` (`local_files_only=True`, fast) |
| Telemetry | OpenTelemetry (traces + metrics + logs) via OTLP/HTTP |

Explicitly **not** used: PyTorch, `sentence-transformers`, FastEmbed, ONNX Runtime.

---

## Files in the baked model dir (`/app/embedder`)

```
openvino_model_qint8_quantized.xml
openvino_model_qint8_quantized.bin
tokenizer.json
tokenizer_config.json
special_tokens_map.json
vocab.txt
```

`embedder.Embedder` hardcodes the `.xml`/`.bin` names. `ci.sh [verify-image]` asserts all four required files exist before the image is pushed.

---

## API

```http
POST /embed
Authorization: Bearer $DENSE_API_KEY
Content-Type: application/json

{"texts": ["..."], "mode": "document" | "query"}
```

```json
{"vectors": [[0.0123, -0.0456, ...], ...]}
```

- Shape `(len(texts), 384)`, `float32`, L2-normalized (`‖v‖ = 1 ± 1e-3`).
- `mode` defaults to `"document"`. Use `"document"` from the indexing pipeline; use `"query"` from the retriever (applies the BGE query prefix server-side).
- `len(texts) ≤ DENSE_MAX_BATCH_SIZE` (default 256) or `400`.
- Auth: `DENSE_API_KEY` unset ⇒ open (local dev only).

`GET /health` returns `backend`, `dim`, `embedder_version`, `threads`, `batch_size`, `max_seq_length`, `otel.*`. `GET /readyz` returns `200` only after model load + warmup.

---

## Configuration

| Var | Default | Meaning |
|---|---|---|
| `DENSE_MODEL_NAME` | `BAAI/bge-small-en-v1.5` | Base model ID |
| `DENSE_DIM` | `384` | Must match model output |
| `DENSE_MAX_BATCH_SIZE` | `256` | HTTP request cap (400 on exceed) |
| `DENSE_BATCH_SIZE` | `32` | OpenVINO static batch dim |
| `DENSE_MAX_SEQ_LENGTH` | `512` | OpenVINO static seq dim |
| `DENSE_THREADS` | `8` | `inference_num_threads` (physical cores) |
| `DENSE_API_KEY` | unset | Bearer auth; unset ⇒ open |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | unset | Enables telemetry |

Set by the Dockerfile, **do not override**: `HF_HOME=/app/embedder`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`.

---

## Identity / fingerprinting

`embedder_version = "bge-small-en-v1.5-openvino-qint8-{max_seq_length}"` — returned by `/health`, recorded in `chunked/_manifest.jsonl` and `pipeline_state`. `HTTPEmbedder` aborts at construction if its local expectation drifts from the server's. Changing `max_seq_length` requires re-running `embed` + `load`.

`Config.service_fingerprint()` = sha256 of `(embedder_version, dense_max_batch_size)`. Ingest writes it; load verifies it before trusting the DB.

---

## Client usage

```python
import httpx

def embed(texts, mode="document"):
    r = httpx.post(
        f"{DENSE_URL}/embed",
        json={"texts": texts, "mode": mode},
        headers={"Authorization": f"Bearer {DENSE_API_KEY}"},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["vectors"]

doc_vecs = embed(chunk_texts, mode="document")   # indexing pipeline
qry_vec   = embed([user_query], mode="query")[0] # retriever
```

Do **not** embed locally with `sentence-transformers`/`fastembed`. Diverging from this endpoint's vector space silently breaks retrieval.

---

## Performance notes

- `max_seq_length` dominates throughput. Set it to your real chunk size (512 for ~500-token chunks). The static reshape happens at container start — no recompile per request.
- `DENSE_BATCH_SIZE` raises throughput and RAM together; sweep with `bench.sh` (`SWEEP_ENV=DENSE_THREADS` or `=DENSE_BATCH_SIZE`).
- `DENSE_THREADS` should equal **physical** cores; SMT logical cores hurt.
- int8 weights are ~35 MB on disk, ~200–400 MB runtime. Steady-state RAM is dominated by OpenVINO, not weights.

---

## Storage

Vectors are `float32`, L2-normalized — cosine and inner product are equivalent. `db.py` stores them as `halfvec(384)` with HNSW `halfvec_cosine_ops`.

```sql
CREATE INDEX idx_chunks_embedding
  ON public.chunks USING hnsw (embedding halfvec_cosine_ops)
  WITH (m = 16, ef_construction = 64);
```

Changing the model revision or `DENSE_MAX_SEQ_LENGTH` invalidates **every** stored vector. Re-embed and reload the corpus; never mix vector spaces in one index.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `/embed` returns `503 model not ready` | Poll `/readyz`, not `/health` |
| `EmbedderDriftError` at ingest start | `DENSE_MAX_SEQ_LENGTH` on dense-service ≠ ingest's `max_seq_length`; align and re-run `embed` |
| Retrieval quality collapses | Client used `mode="document"` for queries — switch to `mode="query"` |
| Startup works but container exits | `HF_HOME`/`EMBEDDER_DIR` overridden at `docker run`; unset |
| Build fails at `hf download` | HF temporarily unreachable, or pinned revision changed; inspect the build log |

---

## Versioning

| Field | Value |
|---|---|
| Base model revision | `BAAI/bge-small-en-v1.5` |
| Artifact revision | `2d7762ea1c5aa69975d766c7b93c15e82eb645ce` |
| Dimensions | 384 (fixed) |
| Max seq (design / recommended) | 512 / 512 |
| Weight precision | int8 (weight-only QDQ) |
| Normalization | L2, always on |

**Re-embed the corpus when:** the model revision changes, dimensions change, pooling/prefix changes, or chunk boundaries shift. **`max_seq_length`, `batch_size`, `threads`, `PERFORMANCE_HINT` are throughput-only and never require re-embedding.**
