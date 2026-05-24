# Onboarding Checklist — Kestral Ticket Triage System

Step-by-step for adding a new service to the system. Every box must be checked before the PR is merged. This ensures every service speaks the same observability language from day one.

---

## Prerequisites

Before writing code, answer these questions and update the relevant contract files:

- [ ] **What is the service name?** Must match `catalogue.md` exactly.
- [ ] **What type?** Entrypoint, middle, or leaf? Determines instrumentation requirements.
- [ ] **What downstream services does it call?** List every outbound HTTP endpoint.
- [ ] **Which calls are business logic?** These propagate trace context.
- [ ] **Which calls are health checks?** These do not propagate trace context.
- [ ] **What is the metrics prefix?** Must be unique across all services.

---

## 1. Service Catalogue

- [ ] Add service row to `catalogue.md` with: name, type, `OTEL_SERVICE_NAME`, metrics prefix, trace/mesh/logs indicators, downstream calls.
- [ ] If leaf service: mark traces and metrics columns with ❌.

---

## 2. Propagation Matrix

- [ ] Add every outbound HTTP call to `propagation_matrix.md`.
- [ ] For each call, specify: caller, callee, endpoint, propagate? (yes/no), method.
- [ ] Health checks must be marked "No" with reason "health check."
- [ ] Business calls must specify propagation method (`inject(headers)` or framework auto-propagation).

---

## 3. Metrics

- [ ] If non-leaf: add service section to `metrics.md`.
- [ ] Define exactly three instruments: `<prefix>.requests`, `<prefix>.duration`, `<prefix>.errors`.
- [ ] Enumerate all label values. Verify total combinations < 100.
- [ ] Define histogram bucket boundaries appropriate for expected latencies.
- [ ] If leaf: confirm no metrics are defined.

---

## 4. Code — OpenTelemetry Setup

### Every Service (Entrypoint, Middle, Leaf)

- [ ] Set `OTEL_SERVICE_NAME` env var in Kubernetes deployment to match `catalogue.md`.
- [ ] Set `OTEL_EXPORTER_OTLP_ENDPOINT` to `http://signoz-otel-collector.signoz.svc.cluster.local:4317`.
- [ ] Set up OTel log bridge at module import time — before any framework import:
  ```python
  from opentelemetry.instrumentation.logging import LoggingInstrumentor, LoggingHandler
  LoggingInstrumentor().instrument()
  ```
- [ ] Suppress noisy third-party loggers (uvicorn, httpx, mcp.server, etc.).
- [ ] Never set root logger to WARNING.
- [ ] Never set `OTEL_TRACES_SAMPLER` or `sampler=` in code.

### Entrypoint and Middle Services

- [ ] Register `TracerProvider` before importing instrumented frameworks:
  ```python
  trace.set_tracer_provider(TracerProvider(...))
  from fastapi import FastAPI  # Or: from fastmcp import FastMCP
  ```
- [ ] Call `FastAPIInstrumentor.instrument_app()` in startup (FastAPI services only).
- [ ] For FastMCP services: no additional instrumentor needed — FastMCP auto-creates spans.
- [ ] Create child spans for every external call with required attributes:
  ```python
  with tracer.start_as_current_span("callee /endpoint") as span:
      span.set_attributes({"http.method": "POST", "http.url": url})
      result = await call()
      span.set_attributes({"http.status_code": 200, "results.count": len(result)})
  ```
- [ ] Create three metric instruments in startup:
  ```python
  request_counter = meter.create_counter("<prefix>.requests", "1", "Total requests")
  request_duration = meter.create_histogram("<prefix>.duration", "s", "Request latency")
  error_counter = meter.create_counter("<prefix>.errors", "1", "Total errors")
  ```
- [ ] Record metrics for every business operation (not health checks).
- [ ] Call `force_flush()` + `shutdown()` on all providers in shutdown handler.
- [ ] Emit log statements inside active spans for trace correlation.

### Leaf Services

- [ ] Set up log bridge only (no TracerProvider, no MeterProvider, no FastAPIInstrumentor).
- [ ] Remove any existing trace/metric code.

---

## 5. Code — Trace Propagation

- [ ] For every business outbound HTTP call: inject trace context.
  ```python
  from opentelemetry.propagate import inject
  headers = {}
  inject(headers)
  await client.post(url, headers=headers)
  ```
- [ ] For every health check HTTP call: do NOT inject trace context.
  ```python
  await client.get(f"{url}/health")  # Bare call
  ```
- [ ] Verify: no `HTTPXClientInstrumentor` usage — it instruments health checks.

---

## 6. Code — Logging

- [ ] Use `log.info()` for request start/completion with key parameters (not bodies).
- [ ] Use `log.warning()` for degraded-but-functional events (retries, fallbacks).
- [ ] Use `log.exception()` for all errors (includes traceback).
- [ ] Never log request bodies, response bodies, PII, or access tokens.
- [ ] Never use DEBUG level in production.
- [ ] Emit debug logs inside active spans:
  ```python
  with tracer.start_as_current_span("operation") as span:
      log.info("Operation started")  # Has trace_id
  ```

---

## 7. Code — Span Attributes

- [ ] Set `http.method`, `http.url` on every CLIENT span before the call.
- [ ] Set `http.status_code` on every CLIENT span after the call.
- [ ] Set `<callee>.results.count` (or equivalent) on every CLIENT span.
- [ ] Use `span.record_exception(exc)` for errors — never set stack trace as attribute.
- [ ] Never set request body, user ID, email, or raw query text as span attributes.

---

## 8. Kubernetes Deployment

- [ ] Set `OTEL_SERVICE_NAME` env var matching `catalogue.md`.
- [ ] Set `OTEL_EXPORTER_OTLP_ENDPOINT` to SigNoz collector.
- [ ] Set `OTEL_EXPORTER_OTLP_INSECURE=true`.
- [ ] Set `DEPLOYMENT_ENVIRONMENT` to `production` (or `staging`).
- [ ] Set `SERVICE_VERSION` to the image tag or git commit SHA.
- [ ] Set `LOG_LEVEL=INFO` (never DEBUG).
- [ ] Define liveness probe: `GET /healthz`.
- [ ] Define readiness probe: `GET /readyz`.

---

## 9. SigNoz Configuration

- [ ] Verify `filter/health` processor excludes `/healthz`, `/readyz`, `/startup` spans.
- [ ] Verify `resource` processor upserts `service.name`.
- [ ] No sampling in collector config during portfolio phase.
- [ ] Add `probabilistic_sampler` when moving to production.

---

## 10. Verification

### Local Testing

- [ ] Start service locally with port-forwarded SigNoz collector.
- [ ] Send a business request.
- [ ] Check SigNoz: service appears in dropdown.
- [ ] Check SigNoz: traces visible with correct service name.
- [ ] Check SigNoz: trace waterfall shows child spans for downstream calls.
- [ ] Check SigNoz: all spans share same `trace_id`.
- [ ] Check SigNoz: metrics visible (filter by service name).
- [ ] Check SigNoz: logs visible with `trace_id` populated.
- [ ] Check SigNoz: click span → "Related Logs" shows correlated logs.
- [ ] Check SigNoz: no `GET /healthz` or `GET /readyz` spans in trace view.

### In-Cluster Testing

- [ ] Deploy to Kubernetes via ArgoCD.
- [ ] Verify pod starts and passes readiness probe.
- [ ] Send a business request through the full chain.
- [ ] Verify full distributed trace in SigNoz (all services in one waterfall).
- [ ] Verify parent-child relationships: `retriever → dense`, `mcp-ops → retriever`, etc.
- [ ] Verify metrics appear under correct service name.
- [ ] Verify logs appear with correct `trace_id`.

---

## 11. Documentation

- [ ] Update `catalogue.md` with the new service row.
- [ ] Update `propagation_matrix.md` with new caller→callee relationships.
- [ ] Update `metrics.md` with new metric definitions.
- [ ] If adding a new alert: update `alerts.md`.
- [ ] If adding a new dashboard panel: update `dashboards.md`.

---

## 12. PR Merge Gates

- [ ] All boxes above checked.
- [ ] Code review verifies invariants 1-12 from `invariants.md`.
- [ ] CI passes (if CI checks for `sampler` keyword, bounded labels, etc.).
- [ ] At least one SigNoz screenshot attached showing: service in dropdown, trace waterfall, correlated logs, metrics.

---

## Quick Reference: Service Types

| Type | Traces | Metrics | Logs | Instruments | Child Spans | Propagation |
|------|--------|---------|------|-------------|-------------|-------------|
| Entrypoint | ✅ SERVER only | ✅ 3 | ✅ | Yes | No (no downstream) | No (leaf of trace) |
| Middle | ✅ SERVER + CLIENT | ✅ 3 | ✅ | Yes | Yes | Yes (inject headers) |
| Leaf | ❌ | ❌ | ✅ | No | No | No |
