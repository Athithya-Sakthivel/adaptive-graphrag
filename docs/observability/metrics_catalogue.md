# Metrics Catalogue — Kestral Ticket Triage System

Every metric name, type, unit, label set, and approved label values across all services. This is the canonical reference — no metric exists unless it's listed here.

---

## Non-Leaf Services (Traces + Metrics + Logs)

### retriever-minimal

| Metric | Type | Unit | Labels | Label Values |
|--------|------|------|--------|-------------|
| `retrieve.requests` | Counter | 1 | `status` | `success`, `client_error`, `server_error` |
| `retrieve.duration` | Histogram | seconds | `status` | `success`, `client_error`, `server_error` |
| `retrieve.errors` | Counter | 1 | `error_type` | `timeout`, `connection_refused`, `dimension_mismatch`, `exception` |

**Histogram buckets:** `[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]` (seconds)

**Cardinality:** 3 status × 4 error types = 12 combinations. Well under the 100 budget.

---
### mcp-server

| Metric | Type | Unit | Labels | Label Values |
|--------|------|------|--------|-------------|
| `mcp_context.requests` | Counter | 1 | `tool`, `status` | `tool`: `lookup_customer`, `get_recent_orders`, `get_order_details`, `check_refund_eligibility`, `search_policies`, `create_ticket`, `escalate_to_human`, `process_auto_refund`<br>`status`: `success`, `client_error`, `server_error` |
| `mcp_context.duration` | Histogram | seconds | `tool` | (same 8 tool values) |
| `mcp_context.errors` | Counter | 1 | `tool`, `error_type` | `tool`: (same 8 values)<br>`error_type`: `timeout`, `connection_refused`, `not_found`, `qdrant_error`, `exception` |

**Histogram buckets:** `[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]` (seconds) — covers database queries (fast) and Qdrant/dense HTTP calls (slower).

**Cardinality:** 8 tools × 3 statuses = 24 for requests; 8 tools × 5 error types = 40 for errors. Total: 64. Well within the 100-label budget.
---

### agent-service

| Metric | Type | Unit | Labels | Label Values |
|--------|------|------|--------|-------------|
| `agent.requests` | Counter | 1 | `node`, `status` | `node`: `classifier`, `auto_resolve`, `human_escalate`, `deflect`, `guardrail`, `context_gatherer`<br>`status`: `success`, `error` |
| `agent.duration` | Histogram | seconds | `node` | (same 6 node values) |
| `agent.errors` | Counter | 1 | `node`, `error_type` | `node`: (same 6 values)<br>`error_type`: `llm_timeout`, `mcp_timeout`, `classification_error`, `exception` |

**Histogram buckets:** `[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]` (seconds) — LLM calls can be slow.

**Cardinality:** 6 nodes × 2 statuses = 12 for requests; 6 nodes × 4 error types = 24 for errors. Total: 36.

---

### frontend-chat

| Metric | Type | Unit | Labels | Label Values |
|--------|------|------|--------|-------------|
| `chat.requests` | Counter | 1 | `status` | `success`, `client_error`, `server_error` |
| `chat.duration` | Histogram | seconds | `status` | `success`, `client_error`, `server_error` |
| `chat.errors` | Counter | 1 | `error_type` | `connection_refused`, `timeout`, `exception` |

**Histogram buckets:** `[0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]` (seconds)

**Cardinality:** 3 status × 3 error types = 9 combinations.

---

### frontend-admin

| Metric | Type | Unit | Labels | Label Values |
|--------|------|------|--------|-------------|
| `admin.requests` | Counter | 1 | `status` | `success`, `client_error`, `server_error` |
| `admin.duration` | Histogram | seconds | `status` | `success`, `client_error`, `server_error` |
| `admin.errors` | Counter | 1 | `error_type` | `connection_refused`, `timeout`, `exception` |

**Histogram buckets:** `[0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]` (seconds)

**Cardinality:** 3 status × 3 error types = 9 combinations.

---

## Leaf Services (Logs Only)

### dense-embedder

No custom metrics. Its latency and error rate are captured by `retriever-minimal`'s `dense /embed` CLIENT span and the `retrieve.duration` histogram.

---

## Cross-Cutting Metric Queries

### Error Rate by Service

```
sum(rate(<prefix>.requests{status="server_error"}[5m])) by (service.name)
```

### P95 Latency by Endpoint

```
histogram_quantile(0.95, sum(rate(<prefix>.duration_bucket[5m])) by (le, service.name))
```

### Error Distribution

```
sum(rate(<prefix>.errors[5m])) by (error_type, service.name)
```