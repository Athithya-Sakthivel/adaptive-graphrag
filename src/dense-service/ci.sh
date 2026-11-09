#!/usr/bin/env bash
# =============================================================================
# src/dense-service/ci.sh
#
# Build -> verify -> push the dense embedding service.
#
# No arguments. Always:
#   - builds with --no-cache
#   - runs the full verification suite
#   - pushes :TAG and :latest to GHCR (fails if GIT_PAT is missing)
#
# Tag format (Kolkata TZ):  YYYY-MM-DD-HH-MM-SS--<git-sha>
#
# Env overrides:
#   GIT_PAT          Required. GHCR token.
#   GHCR_USER        Default: athithya-sakthivel
#   GHCR_REGISTRY    Default: ghcr.io/${GHCR_USER}
#   IMAGE_NAME       Default: dense-embedder
#   DENSE_MODEL_NAME Default: BAAI/bge-small-en-v1.5
#   DENSE_DIM        Default: 384
#   PORT             Default: 9021
#   OBS_PORT         Default: 5081
# =============================================================================

set -Eeuo pipefail
IFS=$'\n\t'

# ---- no-args contract ----
if [[ $# -gt 0 ]]; then
    printf 'ERROR: ci.sh takes no arguments (got: %s)\n' "$*" >&2
    printf 'It always builds with --no-cache and always pushes on success.\n' >&2
    exit 2
fi

# ---- print helpers ----
C_GREEN=$'\033[0;32m'
C_YELLOW=$'\033[1;33m'
C_BLUE=$'\033[0;34m'
C_RED=$'\033[0;31m'
C_RESET=$'\033[0m'

fail() { printf '%sERROR:%s %s\n' "$C_RED" "$C_RESET" "$*" >&2; exit 1; }
log()  { printf '%s==>%s %s\n' "$C_BLUE" "$C_RESET" "$*" >&2; }
pass() { printf '%s[OK]%s %s\n' "$C_GREEN" "$C_RESET" "$*" >&2; }
warn() { printf '%s[WARN]%s %s\n' "$C_YELLOW" "$C_RESET" "$*" >&2; }
step() { printf '\n%s==>%s [%s]\n' "$C_BLUE" "$C_RESET" "$1" >&2; }

# ---- locate service directory (works from any CWD) ----
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---- configuration ----
DENSE_MODEL_NAME="${DENSE_MODEL_NAME:-BAAI/bge-small-en-v1.5}"
DENSE_DIM="${DENSE_DIM:-384}"
PORT="${PORT:-9021}"
OBS_PORT="${OBS_PORT:-5081}"

GHCR_USER="${GHCR_USER:-athithya-sakthivel}"
GHCR_REGISTRY="${GHCR_REGISTRY:-ghcr.io/${GHCR_USER}}"
IMAGE_NAME="${IMAGE_NAME:-dense-embedder}"

NETWORK="dense-test-net"
OBS_NAME="openobserve-dense-test"
DENSE_NAME="dense-service-test"
OBS_USER="root@example.com"
OBS_PASS="Complexpass#123"

GIT_SHA="$(git -C "$SCRIPT_DIR" rev-parse --short HEAD 2>/dev/null || true)"
[[ -n "$GIT_SHA" ]] || GIT_SHA="dev"
BUILD_TIMESTAMP="$(TZ=Asia/Kolkata date +%Y-%m-%d-%H-%M-%S)"
TAG="${BUILD_TIMESTAMP}--${GIT_SHA}"

IMAGE_TAGGED="${GHCR_REGISTRY}/${IMAGE_NAME}:${TAG}"
IMAGE_LATEST="${GHCR_REGISTRY}/${IMAGE_NAME}:latest"

# ---- cleanup on exit (dumps logs on failure) ----
cleanup() {
    local rc=$?
    trap - EXIT

    if [[ $rc -ne 0 ]]; then
        if docker ps -aq --filter "name=${DENSE_NAME}" 2>/dev/null | grep -q .; then
            warn "dense-service logs (tail 120):"
            docker logs --tail 120 "${DENSE_NAME}" 2>&1 | sed 's/^/    /' >&2 || true
        fi
        if docker ps -aq --filter "name=${OBS_NAME}" 2>/dev/null | grep -q .; then
            warn "openobserve logs (tail 40):"
            docker logs --tail 40 "${OBS_NAME}" 2>&1 | sed 's/^/    /' >&2 || true
        fi
    fi

    docker rm -f "${DENSE_NAME}" "${OBS_NAME}" >/dev/null 2>&1 || true
    docker network rm "${NETWORK}" >/dev/null 2>&1 || true

    exit "$rc"
}
trap cleanup EXIT

# ---- phase helpers ----
wait_http() {
    # wait_http <url> <timeout_sec> [user:pass]
    local url="$1" timeout="$2" auth="${3:-}"
    local i=0
    while [[ "$i" -lt "$timeout" ]]; do
        if [[ -n "$auth" ]]; then
            if curl -fsS --max-time 2 -u "$auth" "$url" >/dev/null 2>&1; then
                return 0
            fi
        else
            if curl -fsS --max-time 2 "$url" >/dev/null 2>&1; then
                return 0
            fi
        fi
        i=$((i + 1))
        sleep 1
    done
    return 1
}

run_search() {
    # run_search <type> <sql> <start_us> <end_us>
    local type="$1" sql="$2" start="$3" end="$4" sql_json
    sql_json="$(python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$sql")"
    curl -fsS -u "${OBS_USER}:${OBS_PASS}" \
        -X POST "http://127.0.0.1:${OBS_PORT}/api/default/_search?type=${type}" \
        -H "Content-Type: application/json" \
        -d "{\"query\":{\"sql\":${sql_json},\"start_time\":${start},\"end_time\":${end},\"from\":0,\"size\":50}}"
}

count_hits() {
    python3 -c '
import json, sys
try:
    r = json.load(sys.stdin)
    print(len(r.get("hits", [])))
except Exception:
    print(0)
'
}

# ---- banner ----
printf 'Service dir:    %s\n' "$SCRIPT_DIR"
printf 'Model:          %s (dim=%s)\n' "$DENSE_MODEL_NAME" "$DENSE_DIM"
printf 'Git SHA:        %s\n' "$GIT_SHA"
printf 'Build tag:      %s\n' "$TAG"
printf 'Tagged image:   %s\n' "$IMAGE_TAGGED"
printf 'Latest image:   %s\n' "$IMAGE_LATEST"
printf 'Mode:           build --no-cache, test, push on success\n'

# =============================================================================
# [preflight]
# =============================================================================
step preflight

[[ -d embedder ]] || fail "embedder/ missing -- run ./prepare-model.sh"
[[ -n "$(ls -A embedder 2>/dev/null)" ]] || fail "embedder/ is empty -- run ./prepare-model.sh"
command -v docker  >/dev/null 2>&1 || fail "docker not found"
command -v curl    >/dev/null 2>&1 || fail "curl not found"
command -v python3 >/dev/null 2>&1 || fail "python3 not found"
docker info >/dev/null 2>&1 || fail "docker daemon unavailable"
[[ -n "${GIT_PAT:-}" ]] || fail "GIT_PAT is required (authenticates to GHCR)"

# Detect buildx once; used to pick the builder below.
BUILDX_OK=0
if docker buildx version >/dev/null 2>&1; then
    BUILDX_OK=1
    pass "buildx available (BuildKit enabled)"
else
    warn "buildx not installed; falling back to legacy builder"
    warn "to enable BuildKit: sudo apt install docker-buildx-plugin"
fi

pass "preflight ok"

# =============================================================================
# [build]  -- always --no-cache
# =============================================================================
step build
log "building ${IMAGE_TAGGED} (--no-cache)"

BUILD_ARGS=(
    --no-cache
    --build-arg "DENSE_MODEL_NAME=${DENSE_MODEL_NAME}"
    --build-arg "DENSE_DIM=${DENSE_DIM}"
    --tag "${IMAGE_TAGGED}"
    --tag "${IMAGE_LATEST}"
    --file "${SCRIPT_DIR}/Dockerfile"
)

if [[ "$BUILDX_OK" -eq 1 ]]; then
    DOCKER_BUILDKIT=1 docker build "${BUILD_ARGS[@]}" "$SCRIPT_DIR" \
        || fail "docker build failed (BuildKit)"
else
    DOCKER_BUILDKIT=0 docker build "${BUILD_ARGS[@]}" "$SCRIPT_DIR" \
        || fail "docker build failed (legacy builder). If you see 'buildx component is missing', install: sudo apt install docker-buildx-plugin"
fi

pass "built ${IMAGE_TAGGED}"

# =============================================================================
# [openobserve]
# =============================================================================
step openobserve
docker rm -f "${OBS_NAME}" >/dev/null 2>&1 || true
docker network rm "${NETWORK}" >/dev/null 2>&1 || true
docker network create "${NETWORK}" >/dev/null || fail "docker network create failed"

docker run -d \
    --name "${OBS_NAME}" \
    --network "${NETWORK}" \
    -p "${OBS_PORT}:5080" \
    -e "ZO_ROOT_USER_EMAIL=${OBS_USER}" \
    -e "ZO_ROOT_USER_PASSWORD=${OBS_PASS}" \
    public.ecr.aws/zinclabs/openobserve:latest >/dev/null \
    || fail "openobserve container failed to start"

log "waiting for OpenObserve /healthz"
wait_http "http://127.0.0.1:${OBS_PORT}/healthz" 60 "${OBS_USER}:${OBS_PASS}" \
    || fail "OpenObserve did not become healthy within 60s"
pass "openobserve ready"

# =============================================================================
# [dense]
# =============================================================================
step dense
OBS_AUTH_B64="$(printf '%s' "${OBS_USER}:${OBS_PASS}" | base64 | tr -d '\n')"
docker run -d \
    --name "${DENSE_NAME}" \
    --network "${NETWORK}" \
    -p "${PORT}:8200" \
    -e "OTEL_EXPORTER_OTLP_ENDPOINT=http://${OBS_NAME}:5080/api/default" \
    -e "OTEL_EXPORTER_OTLP_HEADERS=Authorization=Basic ${OBS_AUTH_B64}" \
    -e "OTEL_SERVICE_NAME=dense-embedder-test" \
    -e "OTEL_BSP_SCHEDULE_DELAY=500" \
    -e "OTEL_METRIC_EXPORT_INTERVAL=3000" \
    -e "OTEL_FLUSH_TIMEOUT_MS=5000" \
    "${IMAGE_TAGGED}" >/dev/null \
    || fail "dense-service container failed to start"
pass "dense-service container started"

# =============================================================================
# [ready]
# =============================================================================
step ready
log "waiting for /readyz (up to 120s)"
wait_http "http://127.0.0.1:${PORT}/readyz" 120 \
    || fail "dense-service did not become ready within 120s"
pass "service ready"

# =============================================================================
# [http]
# =============================================================================
step http
log "checking /health"
HEALTH="$(curl -fsS "http://127.0.0.1:${PORT}/health")" || fail "/health request failed"
printf '%s' "$HEALTH" | python3 -c '
import json, sys
h = json.load(sys.stdin)
assert h.get("loaded") is True, "model not loaded"
mp = h.get("model_path")
assert mp and mp.startswith("/app/embedder/"), f"model_path not under /app/embedder: {mp}"
otel = h.get("otel", {})
assert otel.get("initialized") is True, "OTel not initialized"
assert otel.get("trace_provider") is True, "trace provider missing"
assert otel.get("meter_provider") is True, "meter provider missing"
assert otel.get("logger_provider") is True, "logger provider missing"
print("   model_path    =", mp)
print("   otel service  =", otel.get("service"))
print("   otel endpoint =", otel.get("endpoint"))
'
pass "/health contract ok"

log "checking /embed"
EMBED="$(curl -fsS -X POST "http://127.0.0.1:${PORT}/embed" \
    -H 'Content-Type: application/json' \
    -d '{"texts":["hello world","another sentence"]}')" || fail "/embed request failed"
printf '%s' "$EMBED" | DENSE_DIM="$DENSE_DIM" python3 -c '
import json, math, os, sys
dim = int(os.environ["DENSE_DIM"])
v = json.load(sys.stdin)["vectors"]
assert len(v) == 2, f"expected 2 vectors, got {len(v)}"
assert all(len(x) == dim for x in v), f"wrong dim, expected {dim}"
for x in v:
    n = math.sqrt(sum(t * t for t in x))
    assert abs(n - 1.0) < 1e-3, f"vector not L2-normalized (norm={n})"
print("   ", len(v), "vectors x", len(v[0]), "dims, L2-normalized")
'
pass "/embed contract ok"

# =============================================================================
# [telemetry]
# =============================================================================
step telemetry
log "driving extra traffic so metrics and logs have data"
for i in $(seq 1 5); do
    curl -fsS -o /dev/null -X POST "http://127.0.0.1:${PORT}/embed" \
        -H 'Content-Type: application/json' \
        -d "{\"texts\":[\"warmup ${i}\"]}" || true
done

START_US="$(python3 -c 'import time; print(int(time.time()*1_000_000) - 300*1_000_000)')"
END_US="$(python3 -c 'import time; print(int(time.time()*1_000_000) + 60*1_000_000)')"

# traces: hard requirement
TRACE_COUNT=0
TRACES='{}'
for _ in $(seq 1 20); do
    TRACES="$(run_search traces \
        "SELECT service_name, operation_name, duration, span_status FROM \"default\" WHERE service_name = 'dense-embedder-test' ORDER BY _timestamp DESC LIMIT 10" \
        "$START_US" "$END_US" 2>/dev/null)" || TRACES='{}'
    TRACE_COUNT="$(printf '%s' "$TRACES" | count_hits)"
    [[ "$TRACE_COUNT" -gt 0 ]] && break
    sleep 1
done
[[ "$TRACE_COUNT" -gt 0 ]] || fail "no traces from dense-embedder-test after 20s"
pass "traces: ${TRACE_COUNT} row(s)"
printf '%s' "$TRACES" | python3 -c '
import json, sys
r = json.load(sys.stdin)
for h in r["hits"][:3]:
    print("   -", h.get("service_name"), "/", h.get("operation_name"),
          " ", h.get("duration"), "us")
'

# logs: warn only
LOG_COUNT=0
LOGS='{}'
for _ in $(seq 1 15); do
    LOGS="$(run_search logs \
        "SELECT body, severity_text FROM \"default\" ORDER BY _timestamp DESC LIMIT 20" \
        "$START_US" "$END_US" 2>/dev/null)" || LOGS='{}'
    LOG_COUNT="$(printf '%s' "$LOGS" | count_hits)"
    [[ "$LOG_COUNT" -gt 0 ]] && break
    sleep 1
done
if [[ "$LOG_COUNT" -gt 0 ]]; then
    pass "logs: ${LOG_COUNT} row(s)"
    printf '%s' "$LOGS" | python3 -c '
import json, sys
r = json.load(sys.stdin)
for h in r["hits"][:3]:
    body = h.get("body", "")
    if isinstance(body, dict):
        body = body.get("stringValue", str(body))
    print("   -", "[", h.get("severity_text", "?"), "]", str(body)[:100])
' || true
else
    warn "no logs visible after 15s -- LoggingHandler may not be attached"
fi

# metrics: warn only
METRIC_COUNT=0
METRICS='{}'
for _ in $(seq 1 15); do
    METRICS="$(run_search metrics \
        "SELECT __name__, value FROM \"default\" ORDER BY _timestamp DESC LIMIT 20" \
        "$START_US" "$END_US" 2>/dev/null)" || METRICS='{}'
    METRIC_COUNT="$(printf '%s' "$METRICS" | count_hits)"
    [[ "$METRIC_COUNT" -gt 0 ]] && break
    sleep 1
done
if [[ "$METRIC_COUNT" -gt 0 ]]; then
    pass "metrics: ${METRIC_COUNT} row(s)"
    printf '%s' "$METRICS" | python3 -c '
import json, sys
r = json.load(sys.stdin)
for h in r["hits"][:3]:
    print("   -", h.get("__name__"), "=", h.get("value"))
' || true
else
    warn "no metrics visible after 15s -- MeterProvider may not be exporting"
fi

# =============================================================================
# [offline]
# =============================================================================
step offline
log "restarting with --network none (model must load from baked embedder/)"
docker rm -f "${DENSE_NAME}" >/dev/null 2>&1 || true
docker run -d --name "${DENSE_NAME}" --network none "${IMAGE_TAGGED}" >/dev/null \
    || fail "offline container failed to start"

OFFLINE_OK=0
for _ in $(seq 1 120); do
    if docker exec "${DENSE_NAME}" python -c '
import sys, urllib.request as u
try:
    r = u.urlopen("http://127.0.0.1:8200/readyz", timeout=2)
    sys.exit(0 if r.status == 200 else 1)
except Exception:
    sys.exit(1)
' 2>/dev/null; then
        OFFLINE_OK=1
        break
    fi
    sleep 1
done
[[ "$OFFLINE_OK" -eq 1 ]] || fail "service did not become ready with --network none"
pass "model loads with zero network access"

# =============================================================================
# [push]
# =============================================================================
step push
log "authenticating to GHCR as ${GHCR_USER}"
printf '%s' "${GIT_PAT}" | docker login ghcr.io --username "${GHCR_USER}" --password-stdin >/dev/null 2>&1 \
    || fail "GHCR authentication failed -- verify GIT_PAT and package write permissions"
pass "GHCR authenticated"

log "pushing ${IMAGE_TAGGED}"
docker push "${IMAGE_TAGGED}" || fail "push failed: ${IMAGE_TAGGED}"
pass "pushed ${IMAGE_TAGGED}"

log "pushing ${IMAGE_LATEST}"
docker push "${IMAGE_LATEST}" || fail "push failed: ${IMAGE_LATEST}"
pass "pushed ${IMAGE_LATEST}"

# =============================================================================
# summary
# =============================================================================
step summary
printf '\n'
printf '%s\n' '========================================='
printf '  BUILD COMPLETE -- TAG: %s\n' "$TAG"
printf '%s\n' '========================================='
printf '  %s\n' "$IMAGE_TAGGED"
printf '  %s\n' "$IMAGE_LATEST"
printf '%s\n' '========================================='
printf '\n'
printf 'Both tags were pushed to GHCR.\n'
printf '\n'
