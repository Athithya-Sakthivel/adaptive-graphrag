"""Dense embedding service — production grade.

Design contract
---------------
- The model is baked into /app/embedder. ``specific_model_path`` +
  ``local_files_only`` bypass FastEmbed's Hub resolver entirely; the
  process never contacts HuggingFace at runtime.
- Preload is always-on. If the model cannot load from the baked path,
  startup fails and the container exits non-zero.
- CPU parallelism comes from FastEmbed's ``parallel=N`` (N worker
  processes), each with ``threads=1``. This is the *only* reliable way
  to bound ONNX Runtime's intra-op pool — ``ORT_NUM_THREADS`` and
  ``OMP_NUM_THREADS`` env vars are silently ignored by FastEmbed, and
  it otherwise sizes its pool from ``os.cpu_count()`` (which reports
  the host's logical core count in a container). With ``threads=1``,
  set ``parallel`` to the number of *physical* cores.
- Two distinct batch sizes:
    DENSE_MAX_BATCH_SIZE   — HTTP request cap (client-facing, 400 on exceed)
    DENSE_ONNX_CHUNK_SIZE  — FastEmbed internal ONNX chunk size (tuning)
- Three-signal telemetry (traces, metrics, logs) exported via OTLP/HTTP.
  All three providers are created atomically in ``_init_telemetry``; if
  any fails, the service runs without telemetry rather than crashing.
- Graceful shutdown: telemetry flushes are bounded, the executor shuts
  down without blocking, and OTel is flushed even if startup failed.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import numpy as np
from fastapi import FastAPI, HTTPException
from fastembed import TextEmbedding

# OpenTelemetry — traces
# OpenTelemetry — metrics
from opentelemetry import metrics, trace

# OpenTelemetry — logs
from opentelemetry._logs import set_logger_provider
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import ProxyTracerProvider, Status, StatusCode
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=os.getenv("DENSE_LOGLEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("dense")

# ---------------------------------------------------------------------------
# Application configuration
# ---------------------------------------------------------------------------
DENSE_MODEL_NAME = os.getenv("DENSE_MODEL_NAME", "BAAI/bge-small-en-v1.5")
DENSE_DIM = int(os.getenv("DENSE_DIM", "384"))
DENSE_MAX_BATCH_SIZE = int(os.getenv("DENSE_MAX_BATCH_SIZE", "256"))
DENSE_ONNX_CHUNK_SIZE = int(os.getenv("DENSE_ONNX_CHUNK_SIZE", "64"))
DENSE_PARALLEL = int(os.getenv("DENSE_PARALLEL", "8"))
DENSE_NORMALIZE = os.getenv("DENSE_NORMALIZE", "TRUE").upper() in ("1", "TRUE", "YES")
EMBEDDER_DIR = Path(os.getenv("EMBEDDER_DIR", "/app/embedder"))

# ---------------------------------------------------------------------------
# OTel configuration
# ---------------------------------------------------------------------------
OTEL_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
OTEL_HEADERS_RAW = os.getenv("OTEL_EXPORTER_OTLP_HEADERS", "")
OTEL_SERVICE_NAME = os.getenv("OTEL_SERVICE_NAME", "dense-embedder")
OTEL_RESOURCE_ATTRS_RAW = os.getenv("OTEL_RESOURCE_ATTRIBUTES", "")
OTEL_EXPORT_TIMEOUT_S = 10
OTEL_FLUSH_TIMEOUT_MS = int(os.getenv("OTEL_FLUSH_TIMEOUT_MS", "5000"))
OTEL_METRIC_EXPORT_INTERVAL_MS = int(os.getenv("OTEL_METRIC_EXPORT_INTERVAL", "60000"))

# ---------------------------------------------------------------------------
# Runtime state
# ---------------------------------------------------------------------------
_MODEL_LOCK = threading.Lock()
_MODEL: TextEmbedding | None = None
_MODEL_PATH: str | None = None
_READY_AT: float | None = None

_COUNTER_LOCK = threading.Lock()
_REQUESTS = 0
_ERRORS = 0

_TELEMETRY_LOCK = threading.Lock()
_TELEMETRY_INITIALIZED = False

_TRACER_PROVIDER: TracerProvider | None = None
_METER_PROVIDER: MeterProvider | None = None
_METRIC_READER: PeriodicExportingMetricReader | None = None
_LOGGER_PROVIDER: LoggerProvider | None = None

_TRACER: trace.Tracer = trace.get_tracer("dense-embedder")

_REQUEST_COUNTER: metrics.Counter | None = None
_REQUEST_DURATION: metrics.Histogram | None = None
_ERROR_COUNTER: metrics.Counter | None = None
_BATCH_SIZE_HISTOGRAM: metrics.Histogram | None = None
_IN_FLIGHT_GAUGE: metrics.UpDownCounter | None = None

# Single-thread executor: serializes calls into FastEmbed so its internal
# pool is never used concurrently from multiple threads. Actual CPU
# parallelism comes from FastEmbed's `parallel=N` workers.
_INFERENCE_EXECUTOR = ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="inference",
)


# ---------------------------------------------------------------------------
# OTel helpers
# ---------------------------------------------------------------------------
def _parse_kv(s: str) -> dict[str, str]:
    """Parse ``k1=v1,k2=v2`` into a dict. Ignores malformed pairs.

    ``str.partition("=")`` splits on the first ``=`` only, preserving
    base64 padding at the end of the value (e.g. ``...==``).
    """
    out: dict[str, str] = {}
    for pair in (s or "").split(","):
        pair = pair.strip()
        if not pair or "=" not in pair:
            continue
        k, _, v = pair.partition("=")
        out[k.strip()] = v.strip()
    return out


def _normalize_base(base: str) -> str:
    """Validate and trim a base OTLP endpoint URL (no signal path)."""
    base = base.strip()
    parsed = urlparse(base)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError(f"OTLP endpoint must be http(s)://host[:port]/... — got {base!r}")
    return urlunparse(parsed._replace(path=parsed.path.rstrip("/")))


def _signal_endpoint(base: str, signal: str) -> str:
    return f"{base}/v1/{signal}"


def _init_telemetry() -> None:  # noqa: C901  (intentional three-signal setup)
    """Install all three OTel providers atomically. Never raises.

    Tracing, metrics, and logs are all opt-in: if
    ``OTEL_EXPORTER_OTLP_ENDPOINT`` is unset, the service runs without
    telemetry (the default tracer is a no-op). All three providers share
    the same base URL and headers.

    If any provider fails to initialize, the exception is logged and the
    service runs without telemetry rather than refusing to start.
    """
    global _TRACER_PROVIDER, _METER_PROVIDER, _METRIC_READER
    global _LOGGER_PROVIDER, _TRACER, _TELEMETRY_INITIALIZED
    global _REQUEST_COUNTER, _REQUEST_DURATION, _ERROR_COUNTER
    global _BATCH_SIZE_HISTOGRAM, _IN_FLIGHT_GAUGE

    if not OTEL_ENDPOINT:
        log.info("OTel disabled (OTEL_EXPORTER_OTLP_ENDPOINT not set)")
        return

    with _TELEMETRY_LOCK:
        if _TELEMETRY_INITIALIZED:
            return

        current = trace.get_tracer_provider()
        if not isinstance(current, ProxyTracerProvider):
            log.warning("OTel provider already installed; skipping init")
            _TELEMETRY_INITIALIZED = True
            return

        try:
            base = _normalize_base(OTEL_ENDPOINT)
            headers = _parse_kv(OTEL_HEADERS_RAW) or None

            attrs = {
                "service.name": OTEL_SERVICE_NAME,
                "service.version": os.getenv("SERVICE_VERSION", "unknown"),
            }
            attrs.update(_parse_kv(OTEL_RESOURCE_ATTRS_RAW))
            resource = Resource.create(attrs)

            # ---- Traces ----
            trace_provider = TracerProvider(resource=resource)
            trace_provider.add_span_processor(
                BatchSpanProcessor(
                    OTLPSpanExporter(
                        endpoint=_signal_endpoint(base, "traces"),
                        headers=headers,
                        timeout=OTEL_EXPORT_TIMEOUT_S,
                    )
                )
            )
            trace.set_tracer_provider(trace_provider)
            _TRACER_PROVIDER = trace_provider
            _TRACER = trace.get_tracer("dense-embedder")

            # ---- Metrics ----
            reader = PeriodicExportingMetricReader(
                OTLPMetricExporter(
                    endpoint=_signal_endpoint(base, "metrics"),
                    headers=headers,
                    timeout=OTEL_EXPORT_TIMEOUT_S,
                ),
                export_interval_millis=OTEL_METRIC_EXPORT_INTERVAL_MS,
            )
            meter_provider = MeterProvider(
                resource=resource,
                metric_readers=[reader],
            )
            metrics.set_meter_provider(meter_provider)
            _METER_PROVIDER = meter_provider
            _METRIC_READER = reader

            meter = metrics.get_meter("dense-embedder")
            _REQUEST_COUNTER = meter.create_counter(
                "dense_embed_requests_total",
                description="Total /embed requests",
                unit="1",
            )
            _REQUEST_DURATION = meter.create_histogram(
                "dense_embed_duration_seconds",
                description="Embedding request latency",
                unit="s",
            )
            _ERROR_COUNTER = meter.create_counter(
                "dense_embed_errors_total",
                description="Total /embed request errors",
                unit="1",
            )
            _BATCH_SIZE_HISTOGRAM = meter.create_histogram(
                "dense_embed_batch_size",
                description="Number of texts per /embed request",
                unit="1",
            )
            _IN_FLIGHT_GAUGE = meter.create_up_down_counter(
                "dense_embed_in_flight",
                description="In-flight /embed requests",
                unit="1",
            )

            # ---- Logs ----
            logger_provider = LoggerProvider(resource=resource)
            logger_provider.add_log_record_processor(
                BatchLogRecordProcessor(
                    OTLPLogExporter(
                        endpoint=_signal_endpoint(base, "logs"),
                        headers=headers,
                        timeout=OTEL_EXPORT_TIMEOUT_S,
                    )
                )
            )
            set_logger_provider(logger_provider)

            # Guard against duplicate handlers (e.g. re-import, tests).
            root = logging.getLogger()
            if not any(isinstance(h, LoggingHandler) for h in root.handlers):
                root.addHandler(LoggingHandler(logger_provider=logger_provider))
            _LOGGER_PROVIDER = logger_provider

            _TELEMETRY_INITIALIZED = True
            log.info(
                "OTel initialized (3 signals): base=%s service=%s headers=%s",
                base,
                OTEL_SERVICE_NAME,
                "set" if headers else "none",
            )
        except Exception as e:  # noqa: BLE001
            log.exception("OTel init failed; continuing without telemetry: %s", e)


def _shutdown_telemetry() -> None:
    """Flush pending data and shut down all three providers. Idempotent.

    Each step is bounded by ``OTEL_FLUSH_TIMEOUT_MS`` so this function
    cannot hang the shutdown path. Order: metrics reader flush → traces
    flush → shutdown all three providers.
    """
    global _TRACER_PROVIDER, _METER_PROVIDER, _METRIC_READER, _LOGGER_PROVIDER

    if _METRIC_READER is not None:
        try:
            ok = _METRIC_READER.force_flush(timeout_millis=OTEL_FLUSH_TIMEOUT_MS)
            if not ok:
                log.warning("metric reader force_flush timed out")
        except Exception:
            log.exception("metric reader force_flush failed")
        _METRIC_READER = None

    if _TRACER_PROVIDER is not None:
        try:
            ok = _TRACER_PROVIDER.force_flush(timeout_millis=OTEL_FLUSH_TIMEOUT_MS)
            if not ok:
                log.warning("trace provider force_flush timed out")
        except Exception:
            log.exception("trace provider force_flush failed")
        try:
            _TRACER_PROVIDER.shutdown()
        except Exception:
            log.exception("trace provider shutdown failed")
        _TRACER_PROVIDER = None

    if _METER_PROVIDER is not None:
        try:
            _METER_PROVIDER.shutdown()
        except Exception:
            log.exception("meter provider shutdown failed")
        _METER_PROVIDER = None

    if _LOGGER_PROVIDER is not None:
        try:
            _LOGGER_PROVIDER.shutdown()
        except Exception:
            log.exception("logger provider shutdown failed")
        _LOGGER_PROVIDER = None


# ---------------------------------------------------------------------------
# Model resolution and loading
# ---------------------------------------------------------------------------
def _is_snapshot_dir(p: Path) -> bool:
    return (
        p.is_dir()
        and (p / "config.json").exists()
        and any(c.suffix == ".onnx" for c in p.iterdir())
    )


def _resolve_model_path(embedder_dir: Path, model_name: str) -> Path | None:
    """Locate the model snapshot inside ``embedder_dir``.

    FastEmbed maps friendly names (``BAAI/bge-small-en-v1.5``) to
    underlying ONNX repos (``Qdrant/bge-small-en-v1.5-onnx-Q``), so the
    directory name usually does not match ``model_name``. Scan every
    ``models--*/snapshots/*`` directory and pick the newest valid one.
    """
    if not embedder_dir.is_dir():
        return None

    exact = embedder_dir / f"models--{model_name.replace('/', '--')}" / "snapshots"
    if exact.is_dir():
        for snap in sorted(exact.iterdir(), reverse=True):
            if _is_snapshot_dir(snap):
                log.info("model resolved via exact match: %s", snap)
                return snap

    candidates = [
        snap
        for snaps_dir in embedder_dir.glob("models--*/snapshots")
        for snap in sorted(snaps_dir.iterdir(), reverse=True)
        if _is_snapshot_dir(snap)
    ]
    if candidates:
        if len(candidates) > 1:
            log.warning(
                "multiple snapshots under %s — picking %s",
                embedder_dir,
                candidates[0],
            )
        log.info("model resolved via scan: %s", candidates[0])
        return candidates[0]

    raw = embedder_dir / model_name
    if _is_snapshot_dir(raw):
        log.info("model resolved via raw layout: %s", raw)
        return raw

    if _is_snapshot_dir(embedder_dir):
        log.info("model resolved via loose layout: %s", embedder_dir)
        return embedder_dir

    log.error(
        "no valid snapshot under %s (expected models--*/snapshots/<hash>/ "
        "with config.json + *.onnx)",
        embedder_dir,
    )
    return None


def _load_model() -> None:
    """Load the baked model. Raises on any failure — no network fallback."""
    global _MODEL, _MODEL_PATH, _READY_AT
    if _MODEL is not None:
        return
    with _MODEL_LOCK:
        if _MODEL is not None:
            return

        path = _resolve_model_path(EMBEDDER_DIR, DENSE_MODEL_NAME)
        if path is None:
            raise RuntimeError(
                f"model '{DENSE_MODEL_NAME}' not found under {EMBEDDER_DIR}. "
                "Populate embedder/ with ./prepare-model.sh and rebuild."
            )

        _MODEL_PATH = str(path)
        log.info(
            "loading model name=%s path=%s parallel=%d chunk=%d threads=1",
            DENSE_MODEL_NAME,
            _MODEL_PATH,
            DENSE_PARALLEL,
            DENSE_ONNX_CHUNK_SIZE,
        )

        model: TextEmbedding | None = None
        try:
            model = TextEmbedding(
                model_name=DENSE_MODEL_NAME,
                specific_model_path=_MODEL_PATH,
                local_files_only=True,
                parallel=DENSE_PARALLEL,
                batch_size=DENSE_ONNX_CHUNK_SIZE,
                threads=1,
            )
            log.info("model loaded via specific_model_path")
        except Exception as e:
            log.warning(
                "specific_model_path failed (%s: %s); trying cache_dir fallback",
                type(e).__name__,
                e,
            )

        if model is None:
            try:
                model = TextEmbedding(
                    model_name=DENSE_MODEL_NAME,
                    cache_dir=str(EMBEDDER_DIR),
                    local_files_only=True,
                    parallel=DENSE_PARALLEL,
                    batch_size=DENSE_ONNX_CHUNK_SIZE,
                    threads=1,
                )
                log.info("model loaded via cache_dir fallback")
            except Exception as e:
                log.error("cache_dir fallback failed: %s: %s", type(e).__name__, e)
                raise

        # Warm up + verify the model actually produces the expected shape.
        warmup = list(model.embed(["_warmup_"]))
        if not warmup:
            raise RuntimeError("warmup produced no vectors")
        actual_dim = int(np.asarray(warmup[0]).shape[0])
        if actual_dim != DENSE_DIM:
            raise RuntimeError(
                f"model dimension mismatch: configured={DENSE_DIM}, actual={actual_dim}"
            )
        if DENSE_ONNX_CHUNK_SIZE > DENSE_MAX_BATCH_SIZE:
            raise RuntimeError(
                f"DENSE_ONNX_CHUNK_SIZE ({DENSE_ONNX_CHUNK_SIZE}) "
                f"must be <= DENSE_MAX_BATCH_SIZE ({DENSE_MAX_BATCH_SIZE})"
            )

        _MODEL = model
        _READY_AT = time.time()
        log.info(
            "model ready at %s (dim=%d)",
            time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(_READY_AT)),
            actual_dim,
        )


def _do_embed(texts: list[str]) -> list[list[float]]:
    """Run inference. Called from the single-thread executor."""
    model = _MODEL
    if model is None:
        raise RuntimeError("model not loaded")

    out: list[list[float]] = []
    for arr in model.embed(texts):
        v = np.asarray(arr, dtype=np.float32)
        if DENSE_NORMALIZE:
            n = float(np.linalg.norm(v))
            if n > 0.0:
                v = v / n
        if v.shape[0] != DENSE_DIM:
            raise RuntimeError(f"dimension mismatch: expected {DENSE_DIM}, got {v.shape[0]}")
        out.append(v.tolist())

    if len(out) != len(texts):
        raise RuntimeError(f"incomplete output: expected {len(texts)} vectors, got {len(out)}")
    return out


# ---------------------------------------------------------------------------
# Counters (atomic)
# ---------------------------------------------------------------------------
def _bump_request() -> None:
    global _REQUESTS
    with _COUNTER_LOCK:
        _REQUESTS += 1


def _bump_error() -> None:
    global _ERRORS
    with _COUNTER_LOCK:
        _ERRORS += 1


def _snapshot_counters() -> tuple[int, int]:
    with _COUNTER_LOCK:
        return _REQUESTS, _ERRORS


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup ---
    try:
        with _TRACER.start_as_current_span("dense.model.load") as span:
            span.set_attribute("dense.model", DENSE_MODEL_NAME)
            span.set_attribute("dense.dim", DENSE_DIM)
            span.set_attribute("dense.parallel", DENSE_PARALLEL)
            try:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(_INFERENCE_EXECUTOR, _load_model)
                span.set_attribute("dense.model.path", _MODEL_PATH or "")
                span.set_status(Status(StatusCode.OK))
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise RuntimeError("model preload failed") from e
    except Exception:
        # Startup failed — flush telemetry so the error span is exported,
        # then re-raise so the container exits non-zero.
        _shutdown_telemetry()
        raise

    log.info("startup complete")

    # --- Yield to the server ---
    try:
        yield
    finally:
        # --- Shutdown: bounded, ordered ---
        log.info("shutdown: stopping inference executor")
        _INFERENCE_EXECUTOR.shutdown(wait=False, cancel_futures=True)

        log.info("shutdown: flushing telemetry")
        _shutdown_telemetry()

        log.info("shutdown complete")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
_init_telemetry()

app = FastAPI(title="dense-embedder", lifespan=lifespan)
FastAPIInstrumentor.instrument_app(app)


class EmbedRequest(BaseModel):
    texts: list[str]


class EmbedResponse(BaseModel):
    vectors: list[list[float]]


@app.post("/embed", response_model=EmbedResponse)
async def embed(req: EmbedRequest):
    if _MODEL is None:
        raise HTTPException(503, "model not ready")

    if not req.texts:
        raise HTTPException(400, "texts must be a non-empty list")
    if len(req.texts) > DENSE_MAX_BATCH_SIZE:
        raise HTTPException(400, f"batch too large (max={DENSE_MAX_BATCH_SIZE})")

    _bump_request()
    if _IN_FLIGHT_GAUGE is not None:
        _IN_FLIGHT_GAUGE.add(1, {"model": DENSE_MODEL_NAME})

    start = time.perf_counter()
    status_label = "success"

    with _TRACER.start_as_current_span("dense.embed") as span:
        span.set_attribute("dense.batch_size", len(req.texts))
        span.set_attribute("dense.model", DENSE_MODEL_NAME)
        span.set_attribute("dense.normalize", DENSE_NORMALIZE)
        if _BATCH_SIZE_HISTOGRAM is not None:
            _BATCH_SIZE_HISTOGRAM.record(len(req.texts), {"model": DENSE_MODEL_NAME})

        try:
            loop = asyncio.get_running_loop()
            vecs = await asyncio.wait_for(
                loop.run_in_executor(_INFERENCE_EXECUTOR, _do_embed, req.texts),
                timeout=60.0,
            )
            span.set_attribute("dense.vectors_count", len(vecs))
            span.set_status(Status(StatusCode.OK))
            return EmbedResponse(vectors=vecs)

        except TimeoutError:
            status_label = "timeout"
            _bump_error()
            span.set_status(Status(StatusCode.ERROR, "timeout"))
            if _ERROR_COUNTER is not None:
                _ERROR_COUNTER.add(1, {"model": DENSE_MODEL_NAME, "status": "timeout"})
            # Timeout is expected behavior — don't chain it as the cause.
            raise HTTPException(408, "request timed out") from None

        except asyncio.CancelledError:
            status_label = "cancelled"
            span.set_status(Status(StatusCode.ERROR, "cancelled"))
            raise

        except RuntimeError as e:
            status_label = "model_error"
            _bump_error()
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            if _ERROR_COUNTER is not None:
                _ERROR_COUNTER.add(1, {"model": DENSE_MODEL_NAME, "status": "model_error"})
            raise HTTPException(503, "embedding unavailable") from e

        except Exception as e:  # noqa: BLE001
            status_label = "server_error"
            _bump_error()
            log.exception("embed failed")
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            if _ERROR_COUNTER is not None:
                _ERROR_COUNTER.add(1, {"model": DENSE_MODEL_NAME, "status": "server_error"})
            raise HTTPException(500, "internal error") from e

        finally:
            elapsed = time.perf_counter() - start
            if _REQUEST_COUNTER is not None:
                _REQUEST_COUNTER.add(1, {"model": DENSE_MODEL_NAME, "status": status_label})
            if _REQUEST_DURATION is not None:
                _REQUEST_DURATION.record(elapsed, {"model": DENSE_MODEL_NAME})
            if _IN_FLIGHT_GAUGE is not None:
                _IN_FLIGHT_GAUGE.add(-1, {"model": DENSE_MODEL_NAME})


@app.get("/health")
def health():
    requests, errors = _snapshot_counters()
    return {
        "status": "ok",
        "model": DENSE_MODEL_NAME,
        "model_path": _MODEL_PATH,
        "loaded": _MODEL is not None,
        "ready_at": _READY_AT,
        "dim": DENSE_DIM,
        "normalize": DENSE_NORMALIZE,
        "max_batch_size": DENSE_MAX_BATCH_SIZE,
        "onnx_chunk_size": DENSE_ONNX_CHUNK_SIZE,
        "parallel": DENSE_PARALLEL,
        "threads_per_worker": 1,
        "requests": requests,
        "errors": errors,
        "otel": {
            "enabled": bool(OTEL_ENDPOINT),
            "initialized": _TELEMETRY_INITIALIZED,
            "trace_provider": _TRACER_PROVIDER is not None,
            "meter_provider": _METER_PROVIDER is not None,
            "logger_provider": _LOGGER_PROVIDER is not None,
            "endpoint": OTEL_ENDPOINT or None,
            "service": OTEL_SERVICE_NAME,
        },
    }


@app.get("/readyz")
def readyz():
    if _MODEL is None or _READY_AT is None:
        raise HTTPException(503, {"status": "not_ready"})
    return {"status": "ready", "model": DENSE_MODEL_NAME, "ready_at": _READY_AT}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "host_dense:app",
        host=os.getenv("DENSE_HOST", "0.0.0.0"),
        port=int(os.getenv("DENSE_PORT", "8200")),
    )
