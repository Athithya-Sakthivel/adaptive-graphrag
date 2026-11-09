# Dense Embedding Service

A CPU optimized, production-grade HTTP service for generating text embeddings. Built with FastAPI, FastEmbed, and OpenVINO/ONNX Runtime. The service is **stateless, offline-first, and fully self-contained**—the embedding model is baked directly into the image at build time, guaranteeing zero network calls to Hugging Face at runtime.

## Key Features
*   **Offline-First:** Uses `specific_model_path` and `local_files_only=True`. No runtime downloads.
*   **CPU-Optimized:** Tuned for laptops/edge nodes using `parallel=8`, `threads=1`, and `allow_spinning=False` to prevent thermal throttling and OOM kills.
*   **Full Telemetry:** OpenTelemetry traces, metrics, and logs exported via OTLP/HTTP.
*   **Secure:** Runs as a non-root user (`UID 1000`).

---

## API Reference & Embedding I/O

### `POST /embed` – Generate Embeddings

**Input (Request Body):**
```json
{
  "texts": [
    "The quick brown fox jumps over the lazy dog.",
    "A second piece of text to embed."
  ]
}
```
*   **`texts`**: A list of strings. Must be non-empty. The maximum length is capped by `DENSE_MAX_BATCH_SIZE` (default: 256). Sending more will return a `400 Bad Request`.

**Output (Response Body):**
```json
{
  "vectors": [
    [0.0123, -0.0456, 0.0789, ...],
    [-0.0345, 0.0567, -0.0890, ...]
  ]
}
```
*   **`vectors`**: A list of lists of floats.
*   **Shape**: `(len(texts), DENSE_DIM)`.
*   **Dimension**: `384` (for `bge-small-en-v1.5`).
*   **Normalization**: L2-normalized (unit length) by default. If `DENSE_NORMALIZE=FALSE`, raw model logits are returned.
*   **Data Type**: `float32`.

**Error Responses:**
*   `400`: Empty list or batch size exceeded.
*   `503`: Model is still loading or not ready.
*   `500`: Internal inference error.

### `GET /health` – Liveness & Configuration
Returns a JSON snapshot of the service's current state, loaded model path, thread configuration, and OpenTelemetry provider status.

### `GET /readyz` – Readiness Probe
Returns `200 OK` only after the model has fully loaded, compiled its ONNX/OpenVINO graph, and completed a warmup inference.

---

## Configuration

### Embedding Variables
| Variable | Default | Description |
| :--- | :--- | :--- |
| `DENSE_MODEL_NAME` | `BAAI/bge-small-en-v1.5` | Model ID (must match the baked-in model) |
| `DENSE_DIM` | `384` | Must match the model's output dimension |
| `DENSE_MAX_BATCH_SIZE` | `256` | **HTTP cap:** Max texts per `/embed` request |
| `DENSE_ONNX_CHUNK_SIZE` | `32` | **Internal tuning:** Documents per ONNX batch per worker |
| `DENSE_PARALLEL` | `8` | Number of worker processes (set to physical core count) |
| `DENSE_NORMALIZE` | `TRUE` | L2-normalize embeddings to unit length |
| `EMBEDDER_DIR` | `/app/embedder` | Path to the baked-in model files |
| `DENSE_API_KEY` | *(unset)* | If set, requires `Authorization: Bearer <key>` |

### Telemetry Variables (Optional)
| Variable | Default | Description |
| :--- | :--- | :--- |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | *(unset)* | OTLP/HTTP endpoint (e.g., `http://openobserve:5080/api/default`). Tracing is disabled if unset. |
| `OTEL_EXPORTER_OTLP_HEADERS` | *(unset)* | Auth headers (e.g., `Authorization=Basic <b64>`). |
| `OTEL_SERVICE_NAME` | `dense-embedder` | `service.name` attribute in traces/metrics. |

---

## Deployment

```bash
# Example: Running with OpenTelemetry and an API Key
AUTH_B64=$(printf '%s' 'root@example.com:password' | base64)

docker run -d \
  --name dense-service \
  -p 8200:8200 \
  -e DENSE_PARALLEL=8 \
  -e DENSE_ONNX_CHUNK_SIZE=32 \
  -e DENSE_API_KEY="super-secret-key-change-me" \
  -e OTEL_EXPORTER_OTLP_ENDPOINT="http://openobserve:5080/api/default" \
  -e "OTEL_EXPORTER_OTLP_HEADERS=Authorization=Basic ${AUTH_B64}" \
  ghcr.io/athithya-sakthivel/dense-embedder:latest
```

### Observability Note
This service **does not expose a Prometheus `/metrics` endpoint**.
Instead, it emits native OpenTelemetry metrics directly to your OTLP endpoint. This includes:
*   `dense_embed_requests_total` (Counter)
*   `dense_embed_duration_seconds` (Histogram)
*   `dense_embed_in_flight` (UpDownCounter)
*   Traces for `dense.model.load` and `dense.embed`.
*   Logs bridged directly from Python's `logging` module.
