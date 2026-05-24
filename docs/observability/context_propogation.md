# Propagation Matrix — Kestral Ticket Triage System

Which caller propagates trace context to which callee — and which calls
explicitly must not.

---

## Rule

**Propagate** = inject `traceparent` and `tracestate` headers into the
outbound HTTP request. This makes the callee's span a child of the
caller's span.

**Do not propagate** = send a bare HTTP request with no trace headers.
The callee creates an isolated root span.

**Health checks never propagate.** They are infrastructure, not business
logic. Propagating on health checks creates thousands of noise spans per hour.

---

## Matrix

| # | Caller | Callee | Endpoint | Propagate? | Method |
|---|--------|--------|----------|------------|--------|
| 1 | frontend-chat | agent-service | WS `/ws/chat/{session_id}` | Yes | WebSocket upgrade with traceparent header |
| 2 | frontend-admin | agent-service | `POST /tickets/batch` | Yes | `inject(headers)` on HTTP client |
| 3 | frontend-admin | agent-service | `GET /admin/queue` | Yes | `inject(headers)` on HTTP client |
| 4 | agent-service | mcp-server | MCP tools via SSE | Yes | FastMCP client auto-propagates |
| 5 | mcp-server | retriever-minimal | `POST /retrieve` | Yes | `inject(headers)` on HTTP client |
| 6 | retriever-minimal | dense-embedder | `POST /embed` | Yes | `inject(headers)` on HTTP client |
| 7 | retriever-minimal | dense-embedder | `GET /health` | No | Bare HTTP call — health check |
| 8 | retriever-minimal | Qdrant | Vector search | No | Qdrant client handles this internally |
| 9 | mcp-server | PostgreSQL | SQL queries | No | Database protocol, not HTTP |
| 10 | mcp-server | Qdrant | Vector search | No | Qdrant client handles this internally |
| 11 | mcp-server | dense-embedder | `POST /embed` | No | No propagation needed — dense is a leaf service with no traces |
| 12 | All services | * | `GET /healthz` | No | Kubernetes probes — never propagate |
| 13 | All services | * | `GET /readyz` | No | Kubernetes probes — never propagate |

---

## Implementation Patterns

### Business Call — Propagate

```python
from opentelemetry.propagate import inject

async def _embed(text: str) -> list[float]:
    headers: dict[str, str] = {}
    inject(headers)  # Injects traceparent + tracestate

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(
            f"{DENSE_URL}/embed",
            json={"texts": [text]},
            headers=headers,  # Propagated
        )
```

### Health Check — Do Not Propagate

```python
async def _check_dense() -> bool:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{DENSE_URL}/health")
            # No inject(headers) — bare call, no trace context
            return resp.status_code == 200
    except Exception:
        return False
```

### MCP Tool Call — Auto-Propagated

```python
# FastMCP client handles traceparent injection automatically.
# No manual inject() needed — the MCP protocol transports
# trace context in the message metadata.
result = await mcp_client.call_tool("lookup_customer", {"email": email})
```

---

## What a Correct Trace Looks Like

```
POST /retrieve (SERVER, retriever-minimal, trace_id: abc123)
  ├── dense /embed (CLIENT, trace_id: abc123, parent: POST /retrieve)
  │     └── SERVER span on dense-embedder (trace_id: abc123, parent: dense /embed)
  └── qdrant search (CLIENT, trace_id: abc123, parent: POST /retrieve)
```

Every span shares `trace_id: abc123`. The parent-child chain is unbroken.

---

## What a Broken Trace Looks Like

```
POST /retrieve (SERVER, retriever-minimal, trace_id: abc123)
  └── dense /embed (CLIENT, trace_id: abc123, parent: POST /retrieve)

POST /embed (SERVER, dense-embedder, trace_id: xyz789)  <-- Different trace_id!
```

The callee created a new root span because `traceparent` was not sent.
The trace is broken — two separate traces instead of one.

---

## Verification in SigNoz

1. Open any trace from `retriever-minimal`.
2. The waterfall must show child spans for `dense /embed` and `qdrant search`.
3. Click the `dense /embed` span — it must have a corresponding SERVER span on `dense-embedder`.
4. All spans in the waterfall must share the same `trace_id`.
5. No `GET /healthz` or `GET /health` spans should appear in business traces.

---

## New Service Onboarding

When adding a service, update this matrix with:
1. Every outbound HTTP call the service makes
2. Whether each call is business logic (propagate) or health check (do not propagate)
3. The method used (manual `inject()` or framework auto-propagation)

See `onboarding.md` for the complete checklist
