# Pin Python 3.14.8 slim by digest for reproducibility.
FROM docker.io/library/python:3.14.8-slim-trixie@sha256:f85c5697265c178cc6887276c55fe16cf3d14ca35c3df6a5eab3b360534a55d2

ARG DENSE_MODEL_NAME=BAAI/bge-small-en-v1.5
ARG DENSE_DIM=384

# Runtime configuration.
#   DENSE_MAX_BATCH_SIZE   — HTTP request cap (clients must respect this)
#   DENSE_ONNX_CHUNK_SIZE  — FastEmbed internal ONNX chunk size (tuning knob)
#   DENSE_PARALLEL         — FastEmbed worker-process count (~70% of 8 phys cores)
#   OMP/ORT_NUM_THREADS=1  — single-threaded ONNX per worker; parallel does the work
ENV DENSE_MODEL_NAME=${DENSE_MODEL_NAME} \
    DENSE_DIM=${DENSE_DIM} \
    DENSE_MAX_BATCH_SIZE=256 \
    DENSE_ONNX_CHUNK_SIZE=64 \
    DENSE_PARALLEL=6 \
    DENSE_NORMALIZE=TRUE \
    DENSE_HOST=0.0.0.0 \
    DENSE_PORT=8200 \
    EMBEDDER_DIR=/app/embedder \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HOME=/app/embedder \
    FASTEMBED_CACHE=/app/embedder \
    OMP_NUM_THREADS=1 \
    ORT_NUM_THREADS=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Non-root user with fixed uid/gid.
RUN mkdir -p /app /app/embedder \
 && groupadd -g 1000 appgroup \
 && useradd -m -u 1000 -g appgroup -s /bin/sh appuser \
 && chown -R appuser:appgroup /app

# Minimal OS deps. No curl — healthcheck uses Python urllib.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates openssl libssl3 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install pinned deps from requirements.txt. Layer cached independently
# of the (large) embedder COPY below.
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir --upgrade pip wheel \
 && python -m pip install --no-cache-dir -r /app/requirements.txt \
 && python -m pip check

# Bake the model. Fail the build if embedder/ is empty — we would rather
# fail here than ship an image that silently downloads from HF at runtime.
COPY --chown=appuser:appgroup embedder /app/embedder
RUN if [ -z "$(ls -A /app/embedder 2>/dev/null)" ]; then \
      echo "ERROR: embedder/ is empty — run ./prepare-model.sh then rebuild"; \
      exit 1; \
    fi

# Application code last — cheapest to invalidate during dev.
COPY --chown=appuser:appgroup host_dense.py /app/host_dense.py
COPY --chown=appuser:appgroup entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh \
 && chown -R appuser:appgroup /app

USER appuser
ENV HOME=/home/appuser

EXPOSE 8200

# Pure-Python healthcheck. /readyz only returns 200 once the model is loaded.
HEALTHCHECK --interval=15s --timeout=3s --start-period=120s --retries=3 \
  CMD python -c "import sys,urllib.request as u; \
      r=u.urlopen('http://127.0.0.1:8200/readyz',timeout=2); \
      sys.exit(0 if r.status==200 else 1)"

ENTRYPOINT ["sh", "/app/entrypoint.sh"]
