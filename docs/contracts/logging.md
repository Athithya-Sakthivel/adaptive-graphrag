# Logging Contract for OTel-Instrumented Services

**Version:** 1.0.0  
**Status:** Enforceable  
**Applies to:** All services in the ticket-triage-agent system

---

## 1. Principles

1. **Structured only** — No unstructured `print()` or plain `logging.info("string")`. Every log line is a JSON object.
2. **OTel context mandatory** — Every log record carries `trace_id`, `span_id`, and `trace_flags` when inside a span context.
3. **Schema-first** — Log records conform to a documented JSON schema. Unknown fields are silently dropped by the collector.
4. **Levels are semantic** — `ERROR` means human action required; `WARN` means threshold breached but self-healing; `INFO` means business event; `DEBUG` means diagnostic.
5. **No PII in logs** — No emails, phone numbers, API keys, or customer names in any log field. Use `customer_id` (hashed) or `session_id` instead.

---

## 2. Standard Log Record Schema

Every log record emitted by any service MUST conform to this structure:

```json
{
  "timestamp": "2026-05-23T10:30:00.123Z",
  "level": "INFO",
  "service": "dense-embedder",
  "trace_id": "a1b2c3d4e5f67890abcdef1234567890",
  "span_id": "1a2b3c4d5e6f7890",
  "trace_flags": "01",
  "event": "embed.completed",
  "message": "Embedding request completed successfully",
  "attributes": {
    "batch.size": 5,
    "model.name": "BAAI/bge-small-en-v1.5",
    "duration_ms": 45.2,
    "status": "success"
  },
  "error": null
}
```

### Field Definitions

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `timestamp` | string (RFC 3339) | Yes | When the event occurred |
| `level` | string (enum) | Yes | `DEBUG`, `INFO`, `WARN`, `ERROR` |
| `service` | string | Yes | `service.name` from OTel Resource |
| `trace_id` | string (hex, 32 chars) | No | Present when inside a span |
| `span_id` | string (hex, 16 chars) | No | Present when inside a span |
| `trace_flags` | string (hex, 2 chars) | No | `01` = sampled, `00` = not sampled |
| `event` | string (dot.notation) | Yes | Machine-readable event name |
| `message` | string | Yes | Human-readable description |
| `attributes` | object | No | Structured key-value pairs |
| `error` | object or null | No | Present only when `level` is `ERROR` |

### Error Object Schema (when `level: "ERROR"`)

```json
{
  "error": {
    "type": "ModelLoadError",
    "message": "Failed to load model from HuggingFace Hub",
    "stacktrace": "Traceback (most recent call last):\n  File ..."
  }
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `error.type` | string | Yes | Exception class name |
| `error.message` | string | Yes | Exception message (no PII) |
| `error.stacktrace` | string | No | Full stack trace |

---

## 3. Standard Event Names

Event names use dot notation: `<domain>.<action>`. Services MAY add custom events, but MUST use these for common operations:

### HTTP Requests

| Event | Level | When |
|-------|-------|------|
| `http.request.start` | DEBUG | Request received |
| `http.request.completed` | INFO | Response sent |
| `http.request.failed` | ERROR | 5xx or unhandled exception |
| `http.request.validation_failed` | WARN | 4xx from input validation |

### Model/Prediction Operations

| Event | Level | When |
|-------|-------|------|
| `model.load.start` | INFO | Model loading initiated |
| `model.load.completed` | INFO | Model loaded and warmed up |
| `model.load.failed` | ERROR | Model failed to load |
| `model.inference.completed` | DEBUG | Single prediction finished |
| `model.inference.failed` | ERROR | Prediction failed |

### Database Operations

| Event | Level | When |
|-------|-------|------|
| `db.query.slow` | WARN | Query exceeds threshold (default: 100ms) |
| `db.query.failed` | ERROR | Query returned an error |
| `db.connection.pool_exhausted` | WARN | Connection pool at capacity |

### MCP Tool Calls

| Event | Level | When |
|-------|-------|------|
| `mcp.tool.start` | DEBUG | Tool call initiated |
| `mcp.tool.completed` | INFO | Tool call succeeded |
| `mcp.tool.failed` | ERROR | Tool call failed |
| `mcp.tool.timeout` | WARN | Tool call exceeded deadline |

### Service Lifecycle

| Event | Level | When |
|-------|-------|------|
| `service.startup` | INFO | Service finished initialization |
| `service.shutdown` | INFO | Graceful shutdown initiated |
| `service.health_check.failed` | WARN | Health/readiness check returned non-200 |

### DSPy Operations (agent-service only)

| Event | Level | When |
|-------|-------|------|
| `dspy.compile.start` | INFO | DSPy optimization run started |
| `dspy.compile.completed` | INFO | New compiled program saved |
| `dspy.compile.failed` | ERROR | Optimization run failed |
| `dspy.feedback.recorded` | DEBUG | Human override stored for training |

---

## 4. Attribute Conventions

Attributes use lowercase snake_case names with dots for namespacing:

### Common Attributes (all services)

| Attribute | Type | Example | Description |
|-----------|------|---------|-------------|
| `duration_ms` | float | `45.2` | Operation duration in milliseconds |
| `status` | string | `success`, `client_error`, `server_error` | Outcome category |
| `error.type` | string | `RuntimeError` | Exception class (in error object, not attributes) |

### HTTP Attributes

| Attribute | Type | Example |
|-----------|------|---------|
| `http.method` | string | `POST` |
| `http.path` | string | `/embed` |
| `http.status_code` | int | `200` |
| `http.request_size_bytes` | int | `1024` |
| `http.response_size_bytes` | int | `8192` |

### Model Attributes

| Attribute | Type | Example |
|-----------|------|---------|
| `model.name` | string | `BAAI/bge-small-en-v1.5` |
| `model.dim` | int | `384` |
| `model.provider` | string | `CPUExecutionProvider` |
| `batch.size` | int | `32` |
| `vectors.count` | int | `5` |

### Database Attributes

| Attribute | Type | Example |
|-----------|------|---------|
| `db.system` | string | `postgresql` |
| `db.operation` | string | `SELECT` |
| `db.table` | string | `tickets` |
| `db.rows_affected` | int | `1` |
| `db.duration_ms` | float | `12.3` |

---

## 5. Implementation Requirements

### Python Services

Use the standardized `telemetry.py` module (shared library):

```python
from telemetry import get_logger

log = get_logger(__name__)

# Correct — structured event
log.info("Embedding completed", extra={
    "event": "model.inference.completed",
    "attributes": {
        "batch.size": 5,
        "duration_ms": 45.2,
        "status": "success",
    }
})

# WRONG — unstructured string
log.info("Embedding completed in 45.2ms")
```

The `telemetry.py` module MUST:
1. Configure `LoggingHandler` from OTel SDK (bridges Python logging → OTLP)
2. Set a custom `log_record_factory` that injects `trace_id`/`span_id` from current span
3. Validate that `extra` dict contains required `event` field
4. Strip any field names containing `password`, `token`, `secret`, `key`, `authorization`

### Node.js/TypeScript Services (frontends)

```typescript
import { logs } from '@opentelemetry/api';

const logger = logs.getLogger('frontend-chat');

logger.emit({
  severityText: 'INFO',
  body: 'Chat message sent',
  attributes: {
    event: 'chat.message.sent',
    'session.id': sessionId,
    'message.length': messageText.length,
  },
});
```

---

## 6. Validation Rules (Enforced at Collector)

The OTel Collector's `transform` processor enforces these rules:

```yaml
processors:
  transform:
    log_statements:
      - context: log
        statements:
          # Drop logs without event field
          - set(cache["has_event"], attributes["event"]) where attributes["event"] != nil
          - drop() where cache["has_event"] == nil
          
          # Redact PII patterns
          - replace_all_patterns(attributes, "regex", "\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b", "[REDACTED_EMAIL]")
          
          # Enforce timestamp format
          - set(time, Time(attributes["timestamp"], "%Y-%m-%dT%H:%M:%S.%LZ")) where attributes["timestamp"] != nil
```

---

## 7. Examples

### HTTP Request (INFO)

```json
{
  "timestamp": "2026-05-23T10:30:00.123Z",
  "level": "INFO",
  "service": "dense-embedder",
  "trace_id": "a1b2c3d4e5f67890abcdef1234567890",
  "span_id": "1a2b3c4d5e6f7890",
  "trace_flags": "01",
  "event": "http.request.completed",
  "message": "POST /embed completed",
  "attributes": {
    "http.method": "POST",
    "http.path": "/embed",
    "http.status_code": 200,
    "batch.size": 5,
    "duration_ms": 45.2,
    "status": "success"
  },
  "error": null
}
```

### Model Load (INFO)

```json
{
  "timestamp": "2026-05-23T10:29:50.000Z",
  "level": "INFO",
  "service": "dense-embedder",
  "trace_id": null,
  "span_id": null,
  "trace_flags": null,
  "event": "model.load.completed",
  "message": "Model loaded and warmed up",
  "attributes": {
    "model.name": "BAAI/bge-small-en-v1.5",
    "model.dim": 384,
    "model.provider": "CPUExecutionProvider",
    "duration_ms": 2500.0
  },
  "error": null
}
```

### Error (ERROR)

```json
{
  "timestamp": "2026-05-23T10:31:15.456Z",
  "level": "ERROR",
  "service": "dense-embedder",
  "trace_id": "b2c3d4e5f67890abcdef1234567890a1",
  "span_id": "2b3c4d5e6f7890a1",
  "trace_flags": "01",
  "event": "model.inference.failed",
  "message": "Embedding generation failed due to dimension mismatch",
  "attributes": {
    "batch.size": 1,
    "expected_dim": 384,
    "actual_dim": 768,
    "status": "model_error"
  },
  "error": {
    "type": "RuntimeError",
    "message": "Embedding dimension mismatch: expected 384, got 768",
    "stacktrace": "Traceback (most recent call last):\n  File \"host_dense.py\", line 156, in _embed_sync\n    raise RuntimeError(...)\n"
  }
}
```

### Slow Query Warning (WARN)

```json
{
  "timestamp": "2026-05-23T10:32:00.789Z",
  "level": "WARN",
  "service": "mcp-context-server",
  "trace_id": "c3d4e5f67890abcdef1234567890a1b2",
  "span_id": "3c4d5e6f7890a1b2",
  "trace_flags": "01",
  "event": "db.query.slow",
  "message": "Database query exceeded 100ms threshold",
  "attributes": {
    "db.system": "postgresql",
    "db.operation": "SELECT",
    "db.table": "tickets",
    "db.duration_ms": 245.0,
    "db.rows_affected": 12,
    "threshold_ms": 100
  },
  "error": null
}
```

---

## 8. Enforcement Checklist (Per Service)

Before deploying a service, verify:

- [ ] `telemetry.py` (or language equivalent) is imported and initialized before any logging
- [ ] Root logger has `LoggingHandler` attached
- [ ] `log_record_factory` injects `trace_id`/`span_id` automatically
- [ ] All log statements use `extra={"event": "...", "attributes": {...}}`
- [ ] No `print()` statements exist in production code
- [ ] No PII fields in log messages or attributes
- [ ] Error logs include `error.type` and `error.message`
- [ ] Service name is set via `OTEL_SERVICE_NAME` env var
- [ ] OTLP endpoint is configured via `OTEL_EXPORTER_OTLP_ENDPOINT`
- [ ] Log level is configurable via `OTEL_LOG_LEVEL` env var

---

## 9. Appendix: Python `telemetry.py` Reference

This is the shared module every Python service imports. It guarantees schema compliance:

```python
# telemetry.py — NOT included here, imported by all services
#
# Exports:
#   get_logger(name) -> logging.Logger
#   init_telemetry(service_name, otlp_endpoint) -> None
#
# Behavior:
#   - Patches logging.LogRecordFactory to inject trace context
#   - Attaches OTLP LoggingHandler to root logger
#   - Validates 'event' field presence
#   - Strips PII patterns from log messages and attributes
```

Services only need:
```python
from telemetry import get_logger
log = get_logger(__name__)
```

This document is the source of truth for log format across all services. Changes require a version bump and migration plan.