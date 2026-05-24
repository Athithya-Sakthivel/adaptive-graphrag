# Correlated Signals — Kestral Ticket Triage System

How traces, metrics, and logs link together to answer production questions. Every signal is designed to be queried in isolation for breadth and correlated via `trace_id` and matching attribute names for depth.

---

## The Correlation Chain

```
Alert fires (P95 latency > 2s on /retrieve)
  │
  ▼
Metrics: Filter to p95 spike → see it's only for top_k > 50
  │
  ▼
Traces: Filter to top_k > 50 → sample a slow trace
  │
  ▼
Trace View: qdrant search span = 1.8s, dense /embed span = 0.2s
  │
  ▼
Logs (Related Logs tab): "Qdrant search completed results=50" — no error, just slow
  │
  ▼
Action: Add index to Qdrant collection or reduce max top_k
```

Every step is linked by:
- **Same `trace_id`** across all three signals
- **Same attribute names** in metrics labels and span attributes
- **Same service name** in resource attributes

---

## How Each Signal Links

### Traces → Logs

**Mechanism:** `LoggingInstrumentor` automatically injects the current span's `trace_id` and `span_id` into every `log.info()` call made inside an active span context.

**Requirement:** The log statement must be physically inside the `with tracer.start_as_current_span(...)` block. Logs outside spans have no trace context.

**In SigNoz:** Click any span → "Related Logs" tab shows all logs emitted within that span's lifetime.

```
POST /retrieve (trace_id: abc123, span_id: 001)
  ├── log.info("Retrieve started query_len=42")        ← trace_id: abc123, span_id: 001
  ├── dense /embed (trace_id: abc123, span_id: 002)
  │     └── log.info("Embed completed dim=384")        ← trace_id: abc123, span_id: 002
  └── qdrant search (trace_id: abc123, span_id: 003)
        └── log.info("Search done results=5")          ← trace_id: abc123, span_id: 003
```

### Metrics → Traces

**Mechanism:** Metric labels and span attributes use the same key names. Filter a metric spike by label, then search traces by the same attribute.

**In SigNoz:** Metric dashboard shows `retrieve.requests{status="server_error"}` spike → click through → filter traces by `status = server_error` → see individual failing traces.

**Requirement:** Label names must match. See invariant #12.

### Logs → Traces

**Mechanism:** Every log with a `trace_id` can be used to retrieve the full trace.

**In SigNoz:** Log entry shows `trace_id: abc123` → click it → opens the full trace waterfall.

---

## Correlated Debugging Playbooks

### Incident: Service Down

| Step | Signal | Query | What You Learn |
|------|--------|-------|----------------|
| 1 | Metrics | `<caller>.errors{error_type="connection_refused"}` spike | Which upstream caller detected the outage |
| 2 | Traces | Filter by `error_type = connection_refused` | The exact span where the connection failed |
| 3 | Logs | "Related Logs" on the failing span | Last log before crash, exception message |
| 4 | Logs | Service's own logs filtered to the minute before spike | OOM killer, segfault, or panic message |

### Incident: High Latency

| Step | Signal | Query | What You Learn |
|------|--------|-------|----------------|
| 1 | Metrics | `<prefix>.duration` p95 spike, grouped by `status` | Which endpoint is slow, all calls or only errors? |
| 2 | Traces | Filter by endpoint, sort by duration descending | Sample slow traces |
| 3 | Traces | Open trace waterfall | Which child span is the bottleneck? |
| 4 | Logs | "Related Logs" on the slowest child span | Slow query parameters, DNS resolution errors, timeouts |

### Incident: High Error Rate

| Step | Signal | Query | What You Learn |
|------|--------|-------|----------------|
| 1 | Metrics | `<prefix>.errors` spike, grouped by `error_type` | What type of error is spiking? |
| 2 | Traces | Filter by `error_type` and `status = server_error` | Sample failing traces |
| 3 | Traces | Open trace → find the span with `status = ERROR` | Which operation failed? |
| 4 | Logs | "Related Logs" on the ERROR span | Stack trace, validation error message, upstream response |

---

## Signal Matrix per Service

| Service | Trace Root | Child Spans | Metrics | Logs with trace_id |
|---------|-----------|-------------|---------|-------------------|
| frontend-chat | `WS /chat/{session_id}` | `agent-service /ws/chat` | `chat.requests`, `chat.duration`, `chat.errors` | Inside SERVER span only |
| frontend-admin | `GET /admin/queue`, `POST /admin/override` | `agent-service /tickets` | `admin.requests`, `admin.duration`, `admin.errors` | Inside SERVER span only |
| agent-service | `classifier`, `auto_resolve`, `human_escalate` | `mcp-context-server.*`, `mcp-ops-server.*` | `agent.requests`, `agent.duration`, `agent.errors` | Inside each tool span |
| mcp-context-server | FastMCP auto-span per tool | PostgreSQL queries | `mcp_context.requests`, `mcp_context.duration`, `mcp_context.errors` | Inside `record_metrics` decorator |
| mcp-ops-server | FastMCP auto-span per tool | `retriever-minimal /retrieve`, PostgreSQL | `mcp_ops.requests`, `mcp_ops.duration`, `mcp_ops.errors` | Inside `record_metrics` decorator |
| retriever-minimal | `POST /retrieve` | `dense /embed`, `qdrant search` | `retrieve.requests`, `retrieve.duration`, `retrieve.errors` | Inside SERVER span + child spans |
| dense-embedder | — | — | — | No trace_id (leaf service) |

---

## Attributes That Enable Correlation

These attribute names must be identical in both metrics labels and span attributes across all services:

| Attribute | Metrics Label | Span Attribute | Example Values |
|-----------|--------------|----------------|----------------|
| Operation status | `status` | `status` | `success`, `client_error`, `server_error` |
| Error type | `error_type` | `error_type` | `timeout`, `connection_refused`, `validation` |
| Tool name (MCP servers) | `tool` | `tool` | `lookup_customer`, `get_recent_orders` |
| HTTP method | — | `http.method` | `GET`, `POST` |
| HTTP route | — | `http.route` | `/retrieve`, `/embed` |
| HTTP status code | — | `http.status_code` | `200`, `500`, `503` |
| Results count | — | `results.count` | Any integer |

---

## What Not to Correlate

- **Health checks:** No trace propagation, no child spans, no metrics. They exist in isolation.
- **Startup logs:** No span context. They are informational only.
- **Leaf service logs:** No trace context (no spans created). Search by service name and timestamp only.
