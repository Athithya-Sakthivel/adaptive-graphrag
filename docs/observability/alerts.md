# Alerts — Kestral Ticket Triage System

Alert definitions with metric, threshold, duration, and notification channel. Every alert must answer: "What broke? How badly? Who needs to know?" Alerts that don't wake someone up don't ship.

---

## Alert Design Principles

1. **Symptom-based, not cause-based.** Alert on "retrieve error rate > 5%" not "dense service might be down." The symptom tells you what's broken; the trace tells you why.

2. **Actionable only.** Every alert must have a runbook. If the response is "wait and see," it's a dashboard panel, not an alert.

3. **No alerts on individual services for the same symptom.** "High error rate" is one alert applied to all services via label selectors, not seven separate alerts.

4. **Duration before firing.** A single spike is noise. Three consecutive evaluation periods of violation = signal.

---

## Alert Definitions

### 1. High Error Rate

**What:** Error rate exceeds threshold for any non-leaf service.

| Field | Value |
|-------|-------|
| Metric | `sum(rate(<prefix>.requests{status="server_error"}[5m])) by (service.name) / sum(rate(<prefix>.requests[5m])) by (service.name) > 0.05` |
| Threshold | Error rate > 5% |
| Duration | 5 minutes (3 evaluation periods of 2m each) |
| Severity | Critical |
| Runbook | 1. Check SigNoz dashboard for which service is failing. 2. Filter traces by `status = server_error` for that service. 3. Sample failing traces to identify root cause. 4. Check downstream dependencies. |
| Labels | `severity: critical`, `team: platform` |

---

### 2. High Latency — P95

**What:** P95 latency exceeds threshold for any user-facing endpoint.

| Field | Value |
|-------|-------|
| Metric | `histogram_quantile(0.95, sum(rate(<prefix>.duration_bucket[5m])) by (le, service.name)) > <threshold>` |
| Thresholds | `retrieve.duration` > 5s<br>`agent.duration` > 15s (LLM calls are slower)<br>`mcp_context.duration` > 2s (database queries)<br>`mcp_ops.duration` > 5s (includes HTTP to retriever)<br>`chat.duration` > 10s<br>`admin.duration` > 10s |
| Duration | 10 minutes (5 evaluation periods of 2m each) |
| Severity | Warning |
| Runbook | 1. Filter traces for the slow service, sorted by duration descending. 2. Open trace waterfall to identify which child span is the bottleneck. 3. Check "Related Logs" on the slowest span. 4. If LLM calls are slow, check API quota or model availability. |
| Labels | `severity: warning`, `team: platform` |

---

### 3. Service Down — Zero Requests

**What:** A service that should be receiving traffic has zero successful requests.

| Field | Value |
|-------|-------|
| Metric | `sum(rate(<prefix>.requests{status="success"}[5m])) by (service.name) == 0` |
| Threshold | Zero successful requests for 5 minutes |
| Duration | 5 minutes |
| Severity | Critical |
| Runbook | 1. Check Kubernetes pod status (`kubectl get pods -n <namespace>`). 2. Check pod logs for crash loop. 3. Check if upstream caller has connection errors. 4. Check if service is actually deployed (ArgoCD sync status). |
| Labels | `severity: critical`, `team: platform` |
| Exclude | Services not yet deployed (frontend-chat, frontend-admin, agent-service, mcp-ops-server) |

---

### 4. Database Connection Failures

**What:** MCP context server cannot reach PostgreSQL.

| Field | Value |
|-------|-------|
| Metric | `sum(rate(mcp_context.errors{error_type="connection_refused"}[5m])) > 0` |
| Threshold | Any connection refused errors |
| Duration | 5 minutes |
| Severity | Critical |
| Runbook | 1. Check PostgreSQL pod status. 2. Check PgBouncer pod status. 3. Verify connection string and credentials. 4. Check network policies. |
| Labels | `severity: critical`, `team: platform` |

---

### 5. Qdrant Unreachable

**What:** Retriever cannot reach Qdrant.

| Field | Value |
|-------|-------|
| Metric | `sum(rate(retrieve.errors{error_type="connection_refused"}[5m])) > 0` |
| Threshold | Any connection refused errors |
| Duration | 5 minutes |
| Severity | Critical |
| Runbook | 1. Check Qdrant pod status (`kubectl get pods -n qdrant`). 2. Check Qdrant memory usage — may be OOM. 3. Verify collection exists. 4. Verify network policies. |
| Labels | `severity: critical`, `team: platform` |

---

### 6. Dense Service Unreachable

**What:** Retriever cannot reach dense-embedder.

| Field | Value |
|-------|-------|
| Metric | `sum(rate(retrieve.errors{error_type="timeout"}[5m])) > 0` |
| Threshold | Any timeout errors (dense timeouts are the most common failure) |
| Duration | 5 minutes |
| Severity | Critical |
| Runbook | 1. Check dense pod status. 2. Check dense logs for model loading errors. 3. Check if dense is OOM (model loading spikes memory). 4. Verify `DENSE_URL` env var in retriever deployment. |
| Labels | `severity: critical`, `team: platform` |

---

### 7. LLM Call Failures

**What:** Agent service cannot reach the LLM API.

| Field | Value |
|-------|-------|
| Metric | `sum(rate(agent.errors{error_type="llm_timeout"}[5m])) > 0` |
| Threshold | Any LLM timeout errors |
| Duration | 10 minutes (LLM APIs have transient failures — don't alert on single blips) |
| Severity | Warning |
| Runbook | 1. Check LLM API status page (OpenAI/Anthropic). 2. Check API key validity and rate limits. 3. If using local model, check GPU availability. 4. Fallback: agent can escalate to human without LLM. |
| Labels | `severity: warning`, `team: platform` |

---

## Alert Routing

| Severity | Channel | Who |
|----------|---------|-----|
| Critical | PagerDuty / Opsgenie | On-call engineer |
| Warning | Slack `#alerts` channel | Team during business hours |

---

## Silences and Exclusions

| Scenario | Action |
|----------|--------|
| Planned maintenance | Silence all alerts for affected services during maintenance window |
| Service not yet deployed | Exclude from "Service Down" alert via label filter |
| Weekend portfolio demo | Silence non-critical alerts during demo hours |
| First deployment of a service | Expect errors for first 15 minutes — silence during warm-up |

---

## What We Don't Alert On

- **CPU/Memory usage:** Kubernetes handles resource limits. These belong in infrastructure monitoring, not application alerts.
- **Disk usage:** Managed by CloudNative PG and Qdrant operators.
- **Individual 4xx errors:** Client errors are the caller's problem, not ours. Only alert on server errors (5xx).
- **Single-point spikes:** Duration requirement (5+ minutes) filters transient noise.
- **Services not yet deployed:** Alerts exist but are silenced until the service is live.
