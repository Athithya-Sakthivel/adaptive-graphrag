#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'
umask 077

# ------------------------------------------------------------------
#  SigNoz deploy script – no OpAMP, no ArgoCD, single Helm install
#  Works on k3s / EKS.  Usage:  ./signoz.sh [--rollout|--delete]
# ------------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

# ── Configuration (override via environment) ─────────────────────
SIGNOZ_NAMESPACE="${SIGNOZ_NAMESPACE:-signoz}"
SIGNOZ_INFERENCE_NAMESPACE="${SIGNOZ_INFERENCE_NAMESPACE:-inference}"
SIGNOZ_STORAGE_CLASS="${SIGNOZ_STORAGE_CLASS:-default-storage-class}"

HELM_REPO_NAME="${HELM_REPO_NAME:-signoz}"
HELM_REPO_URL="${HELM_REPO_URL:-https://charts.signoz.io}"
HELM_RELEASE="${HELM_RELEASE:-signoz}"
HELM_CHART="${HELM_CHART:-signoz/signoz}"
HELM_VERSION="${HELM_VERSION:-0.126.0}"
HELM_TIMEOUT="${HELM_TIMEOUT:-1h}"
ROLLOUT_TIMEOUT="${ROLLOUT_TIMEOUT:-30m}"

VALUES_FILE="${REPO_ROOT}/src/manifests/signoz/values.yaml"
MANIFESTS_DIR="$(dirname "${VALUES_FILE}")"

TEMP_FILES=()

# ── Helpers ──────────────────────────────────────────────────────
log() { printf '[%s] %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*" >&2; }
fatal() { log "FATAL: $*"; exit 1; }
require_bin() { command -v "$1" >/dev/null 2>&1 || fatal "$1 not found in PATH"; }

cleanup() {
  local f
  for f in "${TEMP_FILES[@]:-}"; do
    rm -f -- "$f" >/dev/null 2>&1 || true
  done
}
trap cleanup EXIT

run_with_timeout() {
  local duration="$1"
  shift
  if command -v timeout >/dev/null 2>&1; then
    timeout --kill-after=10s "$duration" "$@"
  else
    "$@"
  fi
}

new_temp_file() {
  mkdir -p "${MANIFESTS_DIR}"
  local f
  f="$(mktemp "${MANIFESTS_DIR}/.tmp.XXXXXX")"
  TEMP_FILES+=("$f")
  printf '%s' "$f"
}

# ── Delete logic ─────────────────────────────────────────────────
force_delete_namespaced_resources() {
  log "force deleting remaining namespaced resources in ${SIGNOZ_NAMESPACE}"
  while IFS= read -r resource; do
    [[ -n "${resource}" ]] || continue
    while IFS= read -r item; do
      [[ -n "${item}" ]] || continue
      kubectl patch -n "${SIGNOZ_NAMESPACE}" "${item}" --type=merge -p '{"metadata":{"finalizers":[]}}' >/dev/null 2>&1 || true
      kubectl delete -n "${SIGNOZ_NAMESPACE}" "${item}" --grace-period=0 --force --wait=false >/dev/null 2>&1 || true
    done < <(kubectl get "${resource}" -n "${SIGNOZ_NAMESPACE}" -o name 2>/dev/null || true)
  done < <(kubectl api-resources --verbs=list --namespaced -o name 2>/dev/null || true)
}

force_finalize_namespace() {
  local ns_file
  ns_file="$(new_temp_file)"
  if ! kubectl get ns "${SIGNOZ_NAMESPACE}" -o json > "${ns_file}" 2>/dev/null; then
    return 0
  fi
  python3 - "${ns_file}" <<'PY'
from __future__ import annotations
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
obj = json.loads(path.read_text(encoding="utf-8"))
obj.setdefault("spec", {})
obj["spec"]["finalizers"] = []
obj.setdefault("metadata", {})
obj["metadata"].pop("finalizers", None)
path.write_text(json.dumps(obj), encoding="utf-8")
PY
  log "forcing namespace finalization for ${SIGNOZ_NAMESPACE}"
  kubectl replace --raw "/api/v1/namespaces/${SIGNOZ_NAMESPACE}/finalize" -f "${ns_file}" >/dev/null 2>&1 || true
}

delete_all() {
  log "deleting SigNoz release ${HELM_RELEASE} from namespace ${SIGNOZ_NAMESPACE}"
  helm uninstall "${HELM_RELEASE}" -n "${SIGNOZ_NAMESPACE}" >/dev/null 2>&1 || true
  force_delete_namespaced_resources
  kubectl delete ns "${SIGNOZ_NAMESPACE}" --wait=false >/dev/null 2>&1 || true
  kubectl delete clusterrole signoz-otel-collector-signoz --force --grace-period=0 2>/dev/null || true
  kubectl delete clusterrolebinding signoz-otel-collector-signoz --force --grace-period=0 2>/dev/null || true
  sleep 5
  if kubectl get ns "${SIGNOZ_NAMESPACE}" >/dev/null 2>&1; then
    log "namespace still present; finalizing"
    force_finalize_namespace
  fi
  for _ in 1 2 3 4 5; do
    if ! kubectl get ns "${SIGNOZ_NAMESPACE}" >/dev/null 2>&1; then
      rm -f -- "${VALUES_FILE}" >/dev/null 2>&1 || true
      log "deleted release, force-cleaned namespace, and removed generated values file"
      return 0
    fi
    sleep 2
  done
  fatal "namespace '${SIGNOZ_NAMESPACE}' still exists after force deletion"
}

# ── Rollout logic ────────────────────────────────────────────────
generate_password() { python3 -c "import secrets; print(secrets.token_urlsafe(32))"; }

rollout() {
  log "starting SigNoz rollout (no OpAMP)"
  require_bin kubectl; require_bin helm; require_bin python3

  # Pre‑cleanup: delete namespace and cluster‑scoped leftovers
  log "cleaning up any leftover resources…"
  kubectl delete ns "${SIGNOZ_NAMESPACE}" --wait=false >/dev/null 2>&1 || true
  sleep 5
  if kubectl get ns "${SIGNOZ_NAMESPACE}" >/dev/null 2>&1; then
    force_delete_namespaced_resources
    force_finalize_namespace
  fi
  kubectl delete clusterrole signoz-otel-collector-signoz --force --grace-period=0 2>/dev/null || true
  kubectl delete clusterrolebinding signoz-otel-collector-signoz --force --grace-period=0 2>/dev/null || true

  kubectl create namespace "${SIGNOZ_NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -

  SIGNOZ_CLICKHOUSE_PASSWORD="$(generate_password)"
  log "generated new ClickHouse password"

  helm repo add "${HELM_REPO_NAME}" "${HELM_REPO_URL}" --force-update >/dev/null 2>&1 || true
  helm repo update >/dev/null 2>&1

  log "installing/upgrading release ${HELM_RELEASE}"
  run_with_timeout "${HELM_TIMEOUT}" helm upgrade --install "${HELM_RELEASE}" "${HELM_CHART}" \
    --namespace "${SIGNOZ_NAMESPACE}" \
    --create-namespace \
    --version "${HELM_VERSION}" \
    --wait --atomic --timeout "${HELM_TIMEOUT}" \
    --values - <<EOF
global:
  storageClass: "${SIGNOZ_STORAGE_CLASS}"
  clusterName: "kestral"
  cloud: "other"

clickhouse:
  enabled: true
  user: "admin"
  password: "${SIGNOZ_CLICKHOUSE_PASSWORD}"
  layout:
    shardsCount: 1
    replicasCount: 1
  zookeeper:
    enabled: true
    replicaCount: 1
  resources:
    requests: { cpu: "200m", memory: "512Mi" }
    limits:   { cpu: "750m", memory: "1Gi" }
  persistence:
    size: "10Gi"
    storageClass: "${SIGNOZ_STORAGE_CLASS}"
  settings:
    max_server_memory_usage_to_ram_ratio: "0.9"
  profiles:
    default/max_memory_usage: "4000000000"
    admin/max_memory_usage: "4000000000"

signoz:
  replicaCount: 1
  resources:
    requests: { cpu: "100m", memory: "256Mi" }
    limits:   { cpu: "500m", memory: "512Mi" }
  persistence:
    size: "1Gi"
    storageClass: "${SIGNOZ_STORAGE_CLASS}"
  env:
    signoz_telemetrystore_provider: "clickhouse"
    signoz_include_only_log_namespaces: "${SIGNOZ_INFERENCE_NAMESPACE}"
    signoz_emailing_enabled: "false"
    signoz_alertmanager_provider: "signoz"

otelCollector:
  replicaCount: 1
  image:
    registry: docker.io
    repository: signoz/signoz-otel-collector
    tag: v0.144.4
  args:
    - --config=/conf/otel-collector-config.yaml
    - --copy-path=/var/tmp/collector-config.yaml
  ports:
    otlp:
      enabled: true
      servicePort: 4317
      containerPort: 4317
    otlp-http:
      enabled: true
      servicePort: 4318
      containerPort: 4318
  config:
    receivers:
      otlp:
        protocols:
          grpc:
            endpoint: 0.0.0.0:4317
            max_recv_msg_size_mib: 16
          http:
            endpoint: 0.0.0.0:4318
    processors:
      batch:
        send_batch_size: 50000
        timeout: 1s
      filter/health:
        spans:
          exclude:
            match_type: regexp
            span_names:
              - "GET /healthz"
              - "GET /readyz"
              - "GET /health"
              - "GET /startup"
              - "GET /collections"
            attributes:
              - key: http.url
                match_type: regexp
                value: ".*/healthz|.*/readyz|.*/health|.*/collections|.*/ping"
      resource:
        attributes:
          - key: service.name
            from_attribute: service.name
            action: upsert
    exporters:
      clickhousetraces:
        datasource: tcp://\${env:CLICKHOUSE_USER}:\${env:CLICKHOUSE_PASSWORD}@\${env:CLICKHOUSE_HOST}:\${env:CLICKHOUSE_PORT}/\${env:CLICKHOUSE_TRACE_DATABASE}
      signozclickhousemetrics:
        dsn: tcp://\${env:CLICKHOUSE_USER}:\${env:CLICKHOUSE_PASSWORD}@\${env:CLICKHOUSE_HOST}:\${env:CLICKHOUSE_PORT}/\${env:CLICKHOUSE_DATABASE}
      clickhouselogsexporter:
        dsn: tcp://\${env:CLICKHOUSE_USER}:\${env:CLICKHOUSE_PASSWORD}@\${env:CLICKHOUSE_HOST}:\${env:CLICKHOUSE_PORT}/\${env:CLICKHOUSE_LOG_DATABASE}
    service:
      pipelines:
        traces:
          receivers: [otlp]
          processors: [filter/health, resource, batch]
          exporters: [clickhousetraces]
        metrics:
          receivers: [otlp]
          processors: [resource, batch]
          exporters: [signozclickhousemetrics]
        logs:
          receivers: [otlp]
          processors: [resource, batch]
          exporters: [clickhouselogsexporter]
EOF

  log "waiting for rollouts…"
  kubectl -n "${SIGNOZ_NAMESPACE}" rollout status deployment/signoz-otel-collector --timeout="${ROLLOUT_TIMEOUT}"
  kubectl -n "${SIGNOZ_NAMESPACE}" rollout status statefulset/signoz --timeout="${ROLLOUT_TIMEOUT}"
  kubectl -n "${SIGNOZ_NAMESPACE}" rollout status statefulset/chi-signoz-clickhouse-cluster-0-0 --timeout="${ROLLOUT_TIMEOUT}"
  kubectl -n "${SIGNOZ_NAMESPACE}" rollout status statefulset/signoz-zookeeper --timeout="${ROLLOUT_TIMEOUT}"

  CLICKHOUSE_POD="$(kubectl -n "${SIGNOZ_NAMESPACE}" get pods -o name | grep clickhouse | head -1)"
  kubectl -n "${SIGNOZ_NAMESPACE}" exec -it "${CLICKHOUSE_POD}" -- clickhouse-client --query="SELECT 1" >/dev/null || fatal "ClickHouse not healthy"

  kubectl -n "${SIGNOZ_NAMESPACE}" port-forward svc/signoz-otel-collector 4317:4317 &>/tmp/pf-test.log &
  PF_PID=$!
  sleep 3
  if timeout 2 bash -c "echo >/dev/tcp/127.0.0.1/4317" 2>/dev/null; then
    log "Collector gRPC port 4317 is reachable"
  else
    fatal "Collector port 4317 NOT reachable"
  fi
  kill "${PF_PID}" 2>/dev/null || true

  COLLECTOR_SVC="$(kubectl -n "${SIGNOZ_NAMESPACE}" get svc -l app.kubernetes.io/component=otel-collector -o jsonpath='{.items[0].metadata.name}')"
  log ""
  log "==================================="
  log " SigNoz deployment successful!"
  log "==================================="
  log " UI   : kubectl -n ${SIGNOZ_NAMESPACE} port-forward svc/signoz 3301:8080"
  log "         http://localhost:3301"
  log " OTLP : http://${COLLECTOR_SVC}.${SIGNOZ_NAMESPACE}.svc.cluster.local:4317"
  log " ClickHouse password: ${SIGNOZ_CLICKHOUSE_PASSWORD}"
  log "==================================="
}

# ── Main ─────────────────────────────────────────────────────────
main() {
  case "${1:---rollout}" in
    --rollout) rollout ;;
    --delete)  delete_all ;;
    --help|-h)
      cat <<EOF
Usage: signoz.sh [--rollout|--delete]

Environment:
  SIGNOZ_NAMESPACE     (default: signoz)
  SIGNOZ_STORAGE_CLASS (default: default-storage-class)
EOF
      ;;
    *) fatal "unknown option: ${1}" ;;
  esac
}

main "${1:-}"