# OpenInference — Kestral Ticket Triage System

OpenInference extends standard OpenTelemetry spans with AI-specific semantic conventions. It labels every span with an `openinference.span.kind` so SigNoz renders agent traces with the correct icons, waterfall structure, and filtering — no new backend required.

---

## Why OpenInference

| Without OpenInference | With OpenInference |
|----------------------|-------------------|
| Every span looks like `POST /messages` or `GET /mcp` | Spans are labeled `AGENT`, `LLM`, `TOOL`, `GUARDRAIL`, `CHAIN` |
| Can't distinguish a guardrail check from a tool call | Filter by `openinference.span.kind = GUARDRAIL` |
| No token counts, model names, or message contents in traces | `llm.token_count.prompt`, `llm.model_name`, `llm.input_messages` |
| Traces are flat HTTP call lists | Traces show agent reasoning waterfalls with parent-child relationships |

OpenInference sits on top of your existing OTel pipeline. It doesn't replace the collector, exporter, or backend — it just adds attribute names that SigNoz understands.

---

## Span Kinds in This System

| Span Kind | Set By | Where It Appears |
|-----------|--------|-----------------|
| `GUARDRAIL` | Manual in `nodes.py` | `guardrail_classifier` node |
| `AGENT` | Manual in `nodes.py` | `agentic_resolver` root span |
| `LLM` | `DSPyInstrumentor` | Every DSPy `ChainOfThought` call inside TriageProgram and the resolver loop |
| `TOOL` | Manual in `nodes.py` | Every MCP tool call (`search_policies`, `check_refund_eligibility`, etc.) |
| `CHAIN` | Manual in `nodes.py` | `context_gatherer`, `human_escalate` |
| `SERVER` | `FastAPIInstrumentor` (from OTel) | Incoming HTTP requests to agent-service |

---

## Instrumentation Setup

Two instrumentors are registered in `observability.py` during `init_otel()`:

```python
from openinference.instrumentation.langchain import LangChainInstrumentor
from openinference.instrumentation.dspy import DSPyInstrumentor

LangChainInstrumentor().instrument(tracer_provider=_tracer_provider)
DSPyInstrumentor().instrument(tracer_provider=_tracer_provider)
```

- **`LangChainInstrumentor`** — covers LangGraph nodes that use LangChain under the hood. Labels them as `CHAIN` or `AGENT` depending on context.
- **`DSPyInstrumentor`** — covers every `dspy.ChainOfThought` or `dspy.Predict` call. Labels them as `LLM` with `llm.input_messages`, `llm.output_messages`, `llm.model_name`, and `llm.token_count.*`.

Additional span kinds (`GUARDRAIL`, `AGENT`, `TOOL`) are set manually in `nodes.py` using:

```python
span.set_attribute("openinference.span.kind", "GUARDRAIL")
```

---

## What a Full Trace Looks Like

```
SERVER — POST /ws/chat/{session_id} (agent-service, trace_id: abc123)
  └── AGENT: agentic_resolver (2.3s)
        ├── GUARDRAIL: guardrail_classifier (0.4s)
        │     └── LLM: groq/gpt-oss-safeguard-20b
        │           ├── llm.input_messages.0.message.role = "system"
        │           ├── llm.input_messages.0.message.content = "<policy>"
        │           ├── llm.output_messages.0.message.content = "{...}"
        │           └── llm.token_count.prompt = 512, completion = 128
        ├── CHAIN: context_gatherer (0.3s)
        │     ├── TOOL: lookup_customer
        │     │     ├── tool.name = "lookup_customer"
        │     │     └── tool.status = "success"
        │     └── TOOL: get_recent_orders
        ├── LLM: resolver step 0 (0.6s)
        │     └── llm.model_name = "groq/gpt-oss-120b"
        ├── TOOL: search_policies (0.8s)
        │     ├── tool.name = "search_policies"
        │     ├── tool.parameters = {"query": "return policy..."}
        │     └── tool.status = "success"
        ├── TOOL: check_refund_eligibility (0.3s)
        ├── LLM: resolver step 1 (0.4s)
        │     └── llm.model_name = "groq/gpt-oss-120b"
        └── final_answer → response to customer
```

---

## Attributes That Enable Correlation

These OpenInference attribute names match your existing OTel metric labels:

| OpenInference Attribute | Metric Label | Example Values |
|------------------------|-------------|----------------|
| `openinference.span.kind` | — | `AGENT`, `LLM`, `TOOL`, `GUARDRAIL`, `CHAIN` |
| `llm.model_name` | — | `groq/gpt-oss-safeguard-20b` |
| `llm.token_count.prompt` | — | `512` |
| `llm.token_count.completion` | — | `128` |
| `tool.name` | `tool` (in mcp_context metrics) | `search_policies`, `lookup_customer` |
| `tool.status` | `status` (in mcp_context metrics) | `success`, `error` |
| `guardrail.safety` | — | `SAFE`, `UNSAFE` |
| `resolver.steps` | — | `1`–`5` |

---

## Verification in SigNoz

1. Open any trace from `agent-service`.
2. The waterfall must show `GUARDRAIL` → `CHAIN` (context) → `AGENT` (resolver loop).
3. Inside the `AGENT` span, see child `LLM` spans for each resolver step and `TOOL` spans for each MCP call.
4. Click any `LLM` span → see `llm.input_messages`, `llm.output_messages`, token counts.
5. Click any `TOOL` span → see `tool.name`, `tool.parameters`, `tool.status`.
6. Filter traces by `openinference.span.kind = GUARDRAIL` to see only safety decisions.
7. Filter by `openinference.span.kind = LLM` to see all model calls across services.

---

## What OpenInference Does NOT Do

- **Does not replace OTel.** Your existing `TracerProvider`, `MeterProvider`, and OTLP exporter remain unchanged.
- **Does not require a new backend.** Spans still flow to SigNoz via your existing collector.
- **Does not auto-instrument custom nodes.** You must manually set `openinference.span.kind` on nodes like `guardrail_classifier` and `agentic_resolver`.
- **Does not replace metrics or logs.** Your `agent.requests`, `agent.duration`, `agent.errors` metrics and `LoggingInstrumentor` bridge are unchanged.

---

## Installation

Already included in `requirements.txt`:

```
openinference-instrumentation-dspy==0.1.37
openinference-instrumentation-langchain==0.1.66
```

No additional collector configuration needed. OpenInference attributes are standard OTel span attributes and pass through any OTLP-compatible pipeline.