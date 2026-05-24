# Verbosity — Kestral Ticket Triage System

Log levels, required span attributes, and what must never appear in logs or spans. Every line emitted in production must earn its place by answering a specific debugging question.

---

## Log Levels

### INFO — Business Events

What to log:
- Request start and completion with key parameters (not bodies)
- External call results with timing
- State transitions (classification result, routing decision, escalation)

```python
# ✅ Correct
log.info("Retrieve started query_len=%d top_k=%d", len(query), top_k)
log.info("Tool call started: %s", tool_name)
log.info("Classified intent=%s urgency=%d auto_resolvable=%s", intent, urgency, auto_resolvable)

# ❌ Wrong — too verbose
log.info("Entering function _embed_query")
log.info("Request body: %s", json.dumps(request_body))  # PII risk, storage bloat
```

### WARNING — Degraded but Functional

What to log:
- Retry succeeded after initial failure
- Fallback path used
- Near-limit conditions (pool at 80%, retry budget low)

```python
# ✅ Correct
log.warning("Qdrant search retry succeeded after attempt=%d", attempt)
log.warning("Database pool at 80%% capacity active=%d max=%d", active, max_size)

# ❌ Wrong — expected errors
log.warning("Client sent invalid top_k=%d", top_k)  # This is INFO — client error, not our problem
```

### ERROR — Request Failed

What to log:
- Exception with full traceback (use `log.exception`)
- Upstream service unreachable after all retries
- Data corruption or invariant violation

```python
# ✅ Correct
log.exception("Retrieve failed: dense service unreachable")
log.exception("Tool call failed: %s", tool_name)

# ❌ Wrong — using log.error without traceback
log.error("Something broke")  # No traceback — useless for debugging
```

### DEBUG — Never in Production

```python
# Set at module level in every service
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
```

The `LOG_LEVEL` env var allows local debugging. Never set to `DEBUG` in Kubernetes manifests.

---

## Required Span Attributes

### Every SERVER Span

Set automatically by `FastAPIInstrumentor` or FastMCP. No manual work needed.

| Attribute | Source | Example |
|-----------|--------|---------|
| `http.method` | Auto | `POST` |
| `http.route` | Auto | `/retrieve` |
| `http.status_code` | Auto | `200` |

### Every CLIENT Span

Must be set manually after the call completes.

```python
with tracer.start_as_current_span("dense /embed") as span:
    span.set_attributes({
        "http.method": "POST",
        "http.url": f"{DENSE_URL}/embed",
    })
    try:
        vec = await _embed(query)
        span.set_attributes({
            "http.status_code": 200,
            "embedding.dim": len(vec),
        })
    except httpx.TimeoutException:
        span.set_attribute("http.status_code", "timeout")
        span.set_status(Status(StatusCode.ERROR, "Dense timeout"))
        raise
```

| Attribute | Required | Type | Example |
|-----------|----------|------|---------|
| `http.method` | ✅ Yes | string | `"POST"` |
| `http.url` | ✅ Yes | string | `"http://dense-svc:8200/embed"` |
| `http.status_code` | ✅ Yes | string/int | `200`, `"timeout"`, `503` |
| `<callee>.results.count` | ✅ Yes | int | `5`, `0` |
| `error_type` | On failure only | string | `"timeout"`, `"connection_refused"` |

### Every Tool Span (MCP Servers)

Set by `record_metrics` decorator. Additional business attributes set manually.

```python
# In record_metrics decorator:
span.set_attributes({"tool": tool_name, "status": status})

# In tool function if useful:
span.set_attribute("customer.found", customer is not None)
```

---

## Forbidden Content

### Never Log

| Content | Reason |
|---------|--------|
| Request bodies | PII risk (emails, phone numbers, addresses in customer queries) |
| Response bodies | Storage bloat, no debugging value |
| Access tokens or API keys | Security — logs are stored in plaintext |
| Full user profiles | PII risk |
| Full order details | PII risk — log `order_id` instead |
| Raw query text | PII risk + cardinality — log `query.length` instead |

### Never Set as Span Attributes

| Content | Reason |
|---------|--------|
| Request body | PII + cardinality bomb |
| User ID, email, phone | PII |
| Raw query text | Cardinality bomb — use `query.length` |
| Full stack traces | Use `span.record_exception()` — stores trace as event, not attribute |
| IP addresses | PII |
| Session tokens | Security |

---

## Exception Recording

Use `span.record_exception()` — never dump the traceback as a string attribute.

```python
# ✅ Correct
try:
    results = await store.search(vec)
except Exception as exc:
    span.set_status(Status(StatusCode.ERROR, str(exc)))
    span.record_exception(exc)  # Stores structured exception data
    log.exception("Qdrant search failed")  # Log with traceback, correlated via trace_id
    raise

# ❌ Wrong
try:
    results = await store.search(vec)
except Exception as exc:
    span.set_attribute("error.stack", traceback.format_exc())  # String bloat
    raise
```

---

## Third-Party Logger Suppression

Every service must silence noisy libraries. This is the standard block:

```python
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("mcp.server.lowlevel.server").setLevel(logging.WARNING)
logging.getLogger("mcp.server.sse").setLevel(logging.WARNING)
logging.getLogger("sse_starlette.sse").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)
```

Never set the root logger to WARNING — that suppresses your own application logs.

---

## Startup vs Runtime Logs

### Startup (No Span Context)

```python
log.info("OTel logs bridge initialised — endpoint=%s", endpoint)
log.info("OTel traces initialised — endpoint=%s", endpoint)
log.info("Creating database pool (min=%d max=%d)", min_size, max_size)
log.info("Database pool ready")
log.info("Starting mcp-context-server on %s:%s", host, port)
```

These logs have no trace_id. They appear in SigNoz under the service name but are uncorrelated with traces.

### Runtime (Inside Span Context)

```python
log.info("Tool call started: %s", tool_name)   # Has trace_id
log.info("Retrieve completed results=%d", len(results))  # Has trace_id
log.exception("Tool call failed: %s", tool_name)  # Has trace_id
```

These logs have trace_id and span_id. Click any span in SigNoz → "Related Logs" shows these.

---

## Log Format

Every service uses the same format:

```
%(asctime)s [%(name)s] %(levelname)s %(message)s
```

Example output:
```
2026-05-24 09:47:21,234 [retriever-minimal] INFO Retrieve completed query_len=42 results=5
2026-05-24 09:47:21,456 [mcp-context-server] INFO Tool call started: lookup_customer
2026-05-24 09:47:22,789 [mcp-context-server] INFO Tool call completed: lookup_customer (0.123s)
```

No service name in the message itself — that's in `[%(name)s]`. No trace_id in the message — that's injected by `LoggingInstrumentor` and visible in SigNoz.
