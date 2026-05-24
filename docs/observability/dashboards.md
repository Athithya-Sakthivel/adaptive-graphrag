# Dashboards — Kestral Ticket Triage System

Dashboard layouts for SigNoz. One service overview dashboard (filterable by service name), one trace waterfall view, and one business metrics dashboard. No per-service dashboards — the `service.name` variable makes one dashboard cover all services.

---

## Design Principles

1. **One overview dashboard, filterable by `service.name`.** No copy-paste per service.
2. **Every panel answers a specific question.** No "nice to have" charts.
3. **Golden signals first:** throughput, latency, errors. Then breakdowns.
4. **Drill-down enabled:** every chart links to the trace view filtered by the same attributes.

---

## Dashboard 1: Service Overview

**Variable:** `service` — dropdown of all `service.name` values from catalogue.

### Row 1 — Golden Signals

| Panel | Type | Query | Purpose |
|-------|------|-------|---------|
| Request Rate | Time series | `sum(rate(<prefix>.requests[1m])) by (status)` | Throughput — are we receiving traffic? |
| P95 Latency | Time series | `histogram_quantile(0.95, sum(rate(<prefix>.duration_bucket[1m])) by (le))` | Latency — how fast are we responding? |
| Error Rate | Time series | `sum(rate(<prefix>.errors[1m])) by (error_type)` | Errors — what's failing? |

**Layout:** Three panels side by side. 12-hour default range with 1-minute granularity.

### Row 2 — Request Breakdown

| Panel | Type | Query | Purpose |
|-------|------|------|---------|
| Requests by Status | Stacked bar | `sum(rate(<prefix>.requests[5m])) by (status)` | Success vs error ratio |
| Latency Distribution | Heatmap | `histogram_quantile(0.50, ...), (0.95, ...), (0.99, ...)` | Latency profile over time |
| Error Distribution | Pie/Donut | `sum(rate(<prefix>.errors[1h])) by (error_type)` | Which errors dominate? |

### Row 3 — Tool/Endpoint Breakdown (if service has `tool` or `node` label)

| Panel | Type | Query | Purpose |
|-------|------|------|---------|
| Requests by Tool/Node | Stacked bar | `sum(rate(<prefix>.requests[5m])) by (tool)` | Which tool is called most? |
| P95 Latency by Tool/Node | Time series | `histogram_quantile(0.95, sum(rate(<prefix>.duration_bucket[1m])) by (le, tool))` | Which tool is slowest? |
| Errors by Tool/Node | Stacked bar | `sum(rate(<prefix>.errors[5m])) by (tool, error_type)` | Which tool fails most? |

**Applies to:** mcp-context-server, mcp-ops-server, agent-service. Hidden for services without `tool`/`node` label.

---

## Dashboard 2: Trace Waterfall

**Variable:** `trace_id` — populated by clicking through from any metric chart.

### Row 1 — Trace Summary

| Panel | Type | Purpose |
|-------|------|---------|
| Trace Waterfall | Span timeline | Visual breakdown of every span in the trace |
| Span Details | Table | Attributes, duration, status for each span |
| Related Logs | Log table | All logs with this `trace_id` |

### Row 2 — Trace Context

| Panel | Type | Purpose |
|-------|------|---------|
| Service Map | Node graph | Which services participated in this trace? |
| Span Attributes | JSON/Table | Full attribute set for selected span |
| Exception Details | JSON/Table | Stack trace if span has `status = ERROR` |

---

## Dashboard 3: Business Metrics

**Not filterable by service** — this is a cross-cutting view.

### Row 1 — Ticket Triage Outcomes

| Panel | Type | Query | Purpose |
|-------|------|------|---------|
| Tickets by Resolution | Counter | Ticket count by `resolution_type` (auto_resolved, escalated, deflected) | How many tickets does AI resolve vs escalate? |
| Auto-Resolution Rate | Gauge | `auto_resolved / total * 100` | Is the AI getting better? |
| Average Urgency | Gauge | Average `urgency` score per hour | Are customers getting angrier? |

**Data source:** These come from the `tickets` table in PostgreSQL — not from OTel metrics. They require a separate data source in SigNoz or a custom metrics exporter.

### Row 2 — Human Override Impact

| Panel | Type | Query | Purpose |
|-------|------|------|---------|
| Override Count | Time series | Count of `human_overrides` rows per day | How often do humans correct the AI? |
| Accuracy Before/After | Comparison | Classification accuracy before and after DSPy recompile | Does the feedback loop work? |

---

## Dashboard 4: Deployment Markers

Auto-populated from Kubernetes events or ArgoCD sync history. No manual configuration.

| Panel | Type | Purpose |
|-------|------|---------|
| Deploy Timeline | Vertical markers on all time-series charts | Did latency spike after a deploy? |
| Version by Service | Table | Which version of each service is running? |

---

## What We Don't Build

- **Per-service copies of the same dashboard.** The `service` variable handles this.
- **Infrastructure dashboards.** CPU, memory, disk belong in Kubernetes monitoring (Prometheus + Grafana), not application observability (SigNoz).
- **Real-time counters.** SigNoz refreshes every 30 seconds. True real-time belongs in CLI tools (`kubectl top`, `htop`).
- **"Everything" dashboards.** Dashboards with 20+ panels cause scroll fatigue. Seven panels max per dashboard.
- **Custom charts for business metrics that require PostgreSQL queries.** These belong in a separate BI tool (Metabase, Superset) — not SigNoz.

---

## Dashboard Navigation Flow

```
Alert fires (PagerDuty/Slack)
  │
  ▼
Dashboard 1: Service Overview
  │  Filter to affected service
  │  See error rate spike
  │
  ▼
Click error rate chart → Drill down to traces filtered by status=server_error
  │
  ▼
Dashboard 2: Trace Waterfall
  │  Sample a failing trace
  │  See which child span failed
  │  Click "Related Logs"
  │
  ▼
Identify root cause → Fix → Deploy → Check Dashboard 1 for recovery
