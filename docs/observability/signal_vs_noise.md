# Signal vs Noise — Kestral Ticket Triage System

High-signal instrumentation with zero bloat. Every span, metric, and log line must earn its place by answering a specific production question. If it doesn't, it doesn't ship.

---

## The Three Metrics Rule

Every non-leaf service exports exactly three instruments. No exceptions without documented justification.

| Instrument | Type | Unit | Labels | Answers |
|-----------|------|------|--------|---------|
| `<prefix>.requests` | Counter | 1 | `status` | How many requests? What's the error ratio? |
| `<prefix>.duration` | Histogram | seconds | `status` | How fast? What's p50/p95/p99? |
| `<prefix>.errors` | Counter | 1 | `error_type` | What's failing? Timeouts vs validation vs crashes? |

**Why these three:** Throughput, latency, and errors are the three golden signals. Every production question reduces to one of them. Additional metrics create dashboard clutter and alert fatigue without proportional debugging value.

**What we reject:**
- Active connection gauges (pool saturation is visible in duration spikes)
- Cache hit counters (not relevant for this system — no application-level cache)
- Queue depth meters (not relevant — no message queues)
- CPU/memory metrics (Kubernetes handles these; they belong in infra monitoring, not application telemetry)

---

## The Child Span Rule

Every external call gets a child span. Internal logic does not.

| Span Type | When to Create | Example |
|-----------|---------------|---------|
| **SERVER** | Incoming request to your service | `POST /retrieve`, `lookup_customer` tool call |
| **CLIENT** | Outbound call to another service or database | `dense /embed`, `qdrant search`, PostgreSQL query |
| **INTERNAL** | Never — don't create spans for in-process logic | ❌ `_parse_query()`, ❌ `_validate_input()` |

**Why no INTERNAL spans:** They create visual noise in the trace waterfall without actionable information. If a function is slow, it will show up in the CLIENT spans it calls or the SERVER span's total duration. Internal spans are a sign of over-instrumentation.

**What we reject:**
- Spans for every function call
- Spans for loops or iterations
- Spans for data transformation (JSON parsing, validation)
- Spans that are always < 1ms (not actionable)

---

## Logging Levels — What Goes Where

| Level | What to Log | Example | Never Log |
|-------|------------|---------|-----------|
| **INFO** | Request start/completion, external call results | `"Retrieve completed query_len=42 results=5"` | Request bodies, full response payloads |
| **WARNING** | Degraded but functional — retry succeeded, fallback used | `"Qdrant retry succeeded after 1 attempt"` | Expected errors handled gracefully |
| **ERROR** | Request failed — exception caught, returning 5xx | `"Retrieve failed: connection refused"` (with traceback) | Client errors (4xx — those are INFO) |
| **DEBUG** | Never in production | — | Everything |

**Why no DEBUG in production:** DEBUG logs add volume without signal. They're for local development. In production, they increase log storage costs and make it harder to find real ERROR lines.

**What we reject:**
- Logging every request body (PII risk, storage bloat)
- Logging success responses (visible in traces)
- Logging "entering function X" (use spans, not logs)
- Logging stack traces for client errors (4xx is the client's fault, not ours)

---

## Span Attributes — Required vs Forbidden

### Required on Every CLIENT Span

| Attribute | Type | Example | Why |
|-----------|------|---------|-----|
| `http.method` | string | `"POST"` | Identifies the operation |
| `http.url` | string | `"http://dense-svc:8200/embed"` | Identifies the target |
| `http.status_code` | int | `200` | Success or failure |
| `results.count` (or equivalent) | int | `5` | Business-relevant outcome |

### Required on Every SERVER Span

| Attribute | Type | Example | Why |
|-----------|------|---------|-----|
| `http.method` | string | `"POST"` | Set by FastAPIInstrumentor |
| `http.route` | string | `"/retrieve"` | Set by FastAPIInstrumentor |
| `http.status_code` | int | `200` | Set by FastAPIInstrumentor |

### Forbidden on All Spans

| Attribute | Why Forbidden |
|-----------|---------------|
| Request body | PII risk, cardinality bomb, storage bloat |
| Response body | Storage bloat, no debugging value |
| User ID, email, phone | PII risk |
| Raw query text | Cardinality bomb — use `query.length` instead |
| Full stack trace as attribute | Use `span.record_exception()` — it stores the trace in the span event, not as an attribute |

---

## What No Service Ships

- Custom dashboards per service (one set of dashboards covers all services via metric prefix variable)
- Service-specific alerts (alerts are defined once in `alerts.md`, applied via label selectors)
- DEBUG-level logging in production
- More than 3 metric instruments without documented justification
- INTERNAL spans for in-process logic
- Span attributes containing PII or unbounded text
- Metrics with request IDs, user IDs, or query strings as labels
```
