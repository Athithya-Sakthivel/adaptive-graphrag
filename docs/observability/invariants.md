# Observability Invariants — Kestral Ticket Triage System

Rules that must never be broken. Every violation causes a specific, diagnosable production incident. These are enforced by code review, CI checks, and the onboarding checklist — not by convention.

---

## 1. Log Bridge Survives Framework Startup

**Rule:** The OTel log bridge must be attached to the root logger at **module import time** — before any web framework (uvicorn, FastMCP) calls its internal `logging.config.dictConfig()`.

**Why:** Frameworks reset logging configuration during startup, destroying all previously attached handlers. A handler attached at import time survives this wipe. A handler attached in a startup event does not.

**Violation symptom:** Logs appear on stderr but never reach SigNoz. Zero log records in the collector.

**Enforcement:**
```python
# ✅ Correct — main.py, at module level, before any framework import
from opentelemetry.instrumentation.logging import LoggingInstrumentor, LoggingHandler
LoggingInstrumentor().instrument()

# ❌ Wrong — inside FastAPI startup event or FastMCP lifespan
@app.on_event("startup")
async def startup():
    LoggingInstrumentor().instrument()  # Destroyed by uvicorn's dictConfig
```

**Applies to:** All 7 services.

---

## 2. TracerProvider Registered Before Any Library Imports It

**Rule:** `trace.set_tracer_provider()` must execute before any framework (FastAPI, FastMCP) calls `trace.get_tracer()`. Libraries cache the tracer reference at import time or first use.

**Why:** If a library requests a tracer before your provider is registered, it receives a no-op tracer permanently. Re-registering the provider later does not retroactively fix cached references.

**Violation symptom:** Entire service produces zero spans. No errors in logs. Silent failure.

**Enforcement:**
```python
# ✅ Correct — provider first, then import framework
import telemetry
telemetry.init_otel()          # Sets global TracerProvider
from fastmcp import FastMCP    # FastMCP's get_tracer() sees real provider

# ❌ Wrong — framework imported first, got no-op tracer
from fastmcp import FastMCP
telemetry.init_otel()          # Too late — FastMCP already cached no-op
```

**Applies to:** All services that produce traces (entrypoints + middle). Leaf services exempt.

---

## 3. Trace Context Explicitly Propagated — Never on Health Checks

**Rule:** Every outbound HTTP call that is part of a business operation must carry `traceparent` and `tracestate` headers. Health checks, readiness probes, and background ping operations must **never** propagate trace context.

**Why:** Without propagation, downstream services create isolated root spans. The distributed trace is broken — each service appears as a separate trace in SigNoz. Propagating on health checks creates thousands of noise spans.

**Violation symptom:** SigNoz shows each service with its own trace ID for the same request. Health check spans drown real traffic 100:1.

**Enforcement:**
```python
# ✅ Business call — inject context
from opentelemetry.propagate import inject
headers = {}
inject(headers)
await client.post(f"{DENSE_URL}/embed", json=payload, headers=headers)

# ✅ Health check — bare call, no injection
await client.get(f"{DENSE_URL}/health")

# ❌ Wrong — health check with injected context
headers = {}
inject(headers)
await client.get(f"{DENSE_URL}/health", headers=headers)  # NEVER do this
```

**Propagation matrix:** See `propagation_matrix.md` for the complete caller→callee table.

**Applies to:** All services that make outbound HTTP calls.

---

## 4. Every External Call Gets a Child Span with Result Attributes

**Rule:** Any outbound HTTP call or database query within a traced operation must be wrapped in its own span. The span must record the method, URL, status code, and a business-relevant result attribute after the call completes.

**Why:** Without child spans, a 2-second trace shows total duration but no breakdown. The bottleneck is invisible — you can't tell if it was dense, Qdrant, or DNS resolution. Child spans create a timing waterfall that pinpoints the bottleneck in one glance.

**Violation symptom:** Traces with a single flat span. Total duration is visible but debugging requires log correlation or guesswork.

**Enforcement:**
```python
# ✅ Correct — child spans with attributes
with tracer.start_as_current_span("dense /embed") as span:
    span.set_attributes({"http.method": "POST", "http.url": DENSE_URL + "/embed"})
    vec = await _embed(req.query)
    span.set_attributes({"embedding.dim": len(vec), "http.status_code": 200})

with tracer.start_as_current_span("qdrant search") as span:
    span.set_attributes({"http.method": "POST", "collection": COLLECTION_NAME})
    results = await store.search(vec, limit=req.top_k)
    span.set_attributes({"results.count": len(results), "http.status_code": 200})

# ❌ Wrong — flat span, no breakdown
with tracer.start_as_current_span("POST /retrieve") as span:
    vec = await _embed(req.query)      # No child span
    results = await store.search(vec)  # No child span
```

**Required attributes per CLIENT span:**
- `http.method`
- `http.url`
- `http.status_code`
- `<callee>.results.count` (or equivalent business metric)

**Applies to:** All middle services that call external dependencies.

---

## 5. Metric Labels Have Bounded, Known Cardinality

**Rule:** Every label on a counter or histogram must draw from a finite, enumerated set of values defined in `metrics.md`. The total number of unique label combinations per service must be < 100 and known at development time.

**Why:** Each unique label combination creates a new time series in the collector's memory and ClickHouse storage. Unbounded labels (request IDs, user IDs, query strings, free-text) cause memory exhaustion and storage blowup.

**Violation symptom:** SigNoz memory grows unboundedly. Dashboards time out. Collector OOMs on restart.

**Enforcement:**
```python
# ✅ Correct — finite, enumerated values
request_counter.add(1, {"status": "success"})       # 3 values: success, client_error, server_error
request_counter.add(1, {"tool": "lookup_customer"})  # 4 values: one per registered tool

# ❌ Wrong — unbounded cardinality
request_counter.add(1, {"query": user_input})        # Infinite unique values
request_counter.add(1, {"user_id": str(uuid4())})    # Infinite unique values
request_counter.add(1, {"trace_id": trace_id})       # Infinite unique values
```

**Cardinality budget per service:** See `metrics.md` for the approved label sets.

**Applies to:** All services that export custom metrics.

---

## 6. Exactly Three Metric Instruments Per Non-Leaf Service

**Rule:** Every non-leaf service exports exactly three instruments:
- `<prefix>.requests` — Counter, labels: `status`
- `<prefix>.duration` — Histogram (seconds), labels: `status`
- `<prefix>.errors` — Counter, labels: `error_type`

No other metrics without documented justification in `metrics.md`.

**Why:** Three metrics answer every production question: throughput (requests), latency (duration), and error rate (errors). Additional metrics create dashboard clutter and alert fatigue without proportional debugging value.

**Violation symptom:** Dashboards with 20+ metrics per service. Nobody knows which ones matter. Alerts fire on irrelevant signals.

**Enforcement:**
```python
# ✅ Everything you need — three instruments
request_counter = meter.create_counter("retrieve.requests", "1", "Total requests")
request_duration = meter.create_histogram("retrieve.duration", "s", "Request latency")
error_counter = meter.create_counter("retrieve.errors", "1", "Total errors")

# ❌ Bloat — nobody will alert on these
active_connections = meter.create_up_down_counter("pool.connections", "1", ...)
cache_hits = meter.create_counter("cache.hits", "1", ...)
queue_depth = meter.create_gauge("queue.depth", "1", ...)
```

**Approved metric prefixes:** See `catalogue.md` for the service→prefix mapping.

**Applies to:** Entrypoints and middle services. Leaf services exempt.

---

## 7. Sampling Exclusively in the Collector — Never in Application Code

**Rule:** No service sets `OTEL_TRACES_SAMPLER`, passes `sampler=` to `TracerProvider`, or configures any sampling probability. All sampling decisions are made by the SigNoz collector's `probabilistic_sampler` or `tail_sampling` processor.

**Why:** If app A samples at 10% and app B at 100%, traces have gaps — the parent span exists but the child doesn't. Changing sample rate requires redeploying every service. Collector-side sampling is a single configuration point, changeable without redeploys.

**Violation symptom:** Partial traces with gaps. Parent span visible, child span missing. Non-deterministic — depends on which service won the sampling coin flip.

**Enforcement:** CI check — grep for `sampler` in all service code. Fail the build if found.

```yaml
# ✅ Correct — collector config (signoz argo-app)
processors:
  probabilistic_sampler:
    sampling_percentage: 10

# ❌ Wrong — in any service's code
TracerProvider(sampler=TraceIdRatioBased(0.1))  # Never do this
# Or via env var: OTEL_TRACES_SAMPLER=parentbased_traceidratio
```

**Applies to:** All services. No exceptions.

---

## 8. Health Endpoints Excluded from Tracing at the Collector

**Rule:** Kubernetes liveness/readiness probe spans must be dropped by a `filter` processor in the SigNoz collector config. Services must not self-filter.

**Why:** Probes fire every 10 seconds per pod per endpoint. Without filtering, 99% of stored spans are `GET /healthz`. Real traffic is buried. Filtering at the collector is a single configuration point.

**Violation symptom:** SigNoz trace view shows thousands of identical `GET /healthz` spans. Real traces require scrolling past pages of noise.

**Enforcement:** The SigNoz ArgoCD app must include:
```yaml
processors:
  filter/health:
    spans:
      exclude:
        match_type: regexp
        span_names:
          - "GET /healthz"
          - "GET /readyz"
          - "GET /startup"
```

**Applies to:** SigNoz collector configuration. Not per-service.

---

## 9. All Providers Force-Flushed on Graceful Shutdown

**Rule:** `force_flush()` with a 10-second timeout must be called on the trace provider, meter provider, **and** log provider during graceful shutdown. This must happen after all in-flight requests complete.

**Why:** Batch exporters hold telemetry in memory for efficiency. Without flushing, the last N seconds of spans, metrics, and logs are lost on every deploy, scale-down, or restart.

**Violation symptom:** Intermittent data loss. The final spans of a failed request are missing from SigNoz. Non-deterministic — depends on batch timing.

**Enforcement:**
```python
# ✅ Correct — shutdown event in every service
@app.on_event("shutdown")
async def shutdown():
    for provider in [tracer_provider, meter_provider, logger_provider]:
        if provider:
            provider.force_flush(timeout_millis=10_000)
            provider.shutdown()

# ❌ Wrong — no flush, data lost
@app.on_event("shutdown")
async def shutdown():
    pass  # Last 5 seconds of telemetry silently dropped
```

**Applies to:** All services. No exceptions.

---

## 10. Leaf Services Export Logs Only — No Traces, No Custom Metrics

**Rule:** A service that only responds to internal callers and calls nothing downstream (pure compute, no outbound I/O beyond its own response) must export **logs only**. Traces and custom metrics belong in the caller.

**Why:** A leaf service has no downstream calls, so it has nothing to create child spans for. Its latency and error rate are already captured by the caller's CLIENT span. Duplicating traces and metrics creates overlapping, confusing data.

**Violation symptom:** Two spans for the same operation — one SERVER span from the leaf, one CLIENT span from the caller. Confusion about which span is authoritative.

**Leaf services in this system:** `dense-embedder` is the only leaf.

**Non-leaf services:** All other services are entrypoints or middle — they must follow rules 1-9 fully.

**Enforcement:** Leaf services are explicitly tagged in `catalogue.md`. New services must declare their type during onboarding.

---

## 11. Every Log Emitted for Debugging Must Be Inside an Active Span

**Rule:** Log statements that are useful for debugging (request parameters, error details, timing information) must be emitted inside an active span context. Startup logs and health check logs are exempt.

**Why:** Logs inside a span automatically receive the span's `trace_id` and `span_id` from `LoggingInstrumentor`. This enables the "Related Logs" feature in SigNoz — click a span, see its logs. Without this, logs and traces are uncorrelated.

**Violation symptom:** Logs appear in SigNoz but the "Related Logs" tab is empty. No way to jump from a slow span to the log that explains why.

**Enforcement:**
```python
# ✅ Correct — log inside span
with tracer.start_as_current_span("qdrant search") as span:
    log.info("Qdrant search starting top_k=%d", top_k)    # Has trace_id
    results = await store.search(vec, limit=top_k)
    log.info("Qdrant search completed results=%d", len(results))  # Has trace_id

# ❌ Wrong — log outside span
log.info("Starting search")  # No trace_id — uncorrelated
with tracer.start_as_current_span("qdrant search") as span:
    results = await store.search(vec, limit=top_k)
```

**Applies to:** All services that produce traces (entrypoints + middle).

---

## 12. Metric Label Names Match Span Attribute Names

**Rule:** When a metric and a span describe the same dimension (e.g., which endpoint, which status), they must use the same key name. This enables pivoting from a metric spike to a trace sample in SigNoz.

**Why:** SigNoz allows filtering traces by the same attributes used in metric labels. If names don't match, you can't drill down from an error rate spike to individual failing traces.

**Violation symptom:** Metric shows error rate spike for `endpoint=/retrieve`. Traces use `http.route` instead. Can't filter traces to find the failing ones.

**Enforcement:**
```python
# ✅ Correct — consistent names
# In metrics:
request_counter.add(1, {"status": "server_error"})
# In spans:
span.set_attribute("status", "server_error")

# ❌ Wrong — mismatched names
# In metrics:
request_counter.add(1, {"http_status": "500"})
# In spans:
span.set_attribute("http.status_code", 500)
```

**Standardized attributes:** See `metrics.md` and `verbosity.md` for the canonical key names.

**Applies to:** All services that export both metrics and traces.

---

## Enforcement Summary

| # | Invariant | Enforced By |
|---|-----------|-------------|
| 1 | Log bridge at import time | Code review — check import location |
| 2 | TracerProvider before imports | Code review — check import order |
| 3 | Explicit propagation, not on health checks | Code review — every HTTP call must state intent |
| 4 | Child spans for external calls | Code review — every external call wrapped |
| 5 | Bounded cardinality | CI check — grep for dynamic label values |
| 6 | Three metrics per service | Code review — no extra instruments without justification |
| 7 | Collector-side sampling | CI check — grep for `sampler` in codebase |
| 8 | Health checks filtered at collector | ArgoCD app is source of truth |
| 9 | Graceful shutdown flush | Code review — every service has shutdown handler |
| 10 | Leaf services: logs only | `catalogue.md` type field |
| 11 | Debug logs inside spans | Code review — log placement relative to span |
| 12 | Matching metric/span attribute names | `metrics.md` + `verbosity.md` are canonical |