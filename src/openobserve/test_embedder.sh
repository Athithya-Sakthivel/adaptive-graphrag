#!/usr/bin/env bash
# test_embedder.sh — Verify that OpenObserve ingests OTLP traces.
#
# Behaviour:
#   1. If an OpenObserve is already reachable at $PORT, reuse it (no spawn).
#   2. Otherwise, start a fresh container on the first free port in
#      ${PORT}..$((PORT+4)), record its name, and clean it up on exit.
#   3. Send a synthetic OTLP/HTTP trace payload.
#   4. Poll the search API until the trace appears (or time out).
#   5. Print stream inventory so you can inspect the result.
#
# Usage:
#   ./test_embedder.sh                 # run (reuse or spawn)
#   ./test_embedder.sh --teardown      # remove test-spawned containers only
#   PORT=5081 ./test_embedder.sh       # prefer a different port
#
# Env overrides:
#   PORT      (default: 5080)
#   EMAIL     (default: root@example.com)
#   PASSWORD  (default: Complexpass#123)
#   KEEP=1    (don't tear down the container the script spawned)

set -euo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-5080}"
EMAIL="${EMAIL:-root@example.com}"
PASSWORD="${PASSWORD:-Complexpass#123}"
KEEP="${KEEP:-0}"
AUTH="${EMAIL}:${PASSWORD}"
SPAWNED_CONTAINER=""    # set only if we start our own
SPAWNED_PORT=""         # record where we started it

teardown_spawned() {
  if [ -n "${SPAWNED_CONTAINER}" ] && [ "${KEEP}" != "1" ]; then
    echo "[teardown] Removing ${SPAWNED_CONTAINER}"
    docker rm -f "${SPAWNED_CONTAINER}" >/dev/null 2>&1 || true
  fi
}
trap teardown_spawned EXIT

# ---------------------------------------------------------------- --teardown
if [ "${1:-}" = "--teardown" ]; then
  echo "[teardown] Removing any test-spawned containers"
  docker ps -aq -f "name=^/openobserve-embedder-test" | xargs -r docker rm -f >/dev/null 2>&1 || true
  echo "[teardown] Done. (Containers started by setup.sh are left alone.)"
  exit 0
fi

# ---------------------------------------------------------------- helpers
probe() {
  # Returns 0 if ${1} responds 200 on /healthz
  local p="${1}"
  curl -fsS --max-time 2 -u "${AUTH}" "http://localhost:${p}/healthz" >/dev/null 2>&1
}

port_busy() {
  # Returns 0 if anything is listening on ${1} (even unhealthy)
  local p="${1}"
  (exec 3<>"/dev/tcp/127.0.0.1/${p}") 2>/dev/null && exec 3>&- && return 0
  return 1
}

start_obs() {
  # start_obs <port> <container-name>
  local p="${1}" name="${2}"
  echo "[start] Spawning ${name} on :${p}"
  if ! docker run -d \
      --name "${name}" \
      -p "${p}:5080" \
      -v "$PWD/data:/data" \
      -e ZO_DATA_DIR=/data \
      -e ZO_ROOT_USER_EMAIL="${EMAIL}" \
      -e ZO_ROOT_USER_PASSWORD="${PASSWORD}" \
      public.ecr.aws/zinclabs/openobserve:latest >/dev/null; then
    echo "[FAIL] docker run failed for ${name} on port ${p}"
    return 1
  fi
  return 0
}

# ---------------------------------------------------------------- 1. reuse or spawn
echo "[1/5] Locating OpenObserve"

CHOSEN_PORT=""
if probe "${PORT}"; then
  echo "       Reusing existing OpenObserve at http://localhost:${PORT}"
  CHOSEN_PORT="${PORT}"
else
  # Try PORT..PORT+4 for a free port, skipping busy-but-unhealthy ones.
  for offset in 0 1 2 3 4; do
    p=$((PORT + offset))
    if probe "${p}"; then
      echo "       Reusing existing OpenObserve at http://localhost:${p}"
      CHOSEN_PORT="${p}"
      break
    fi
    if port_busy "${p}"; then
      echo "       Port ${p} is busy but not an OpenObserve — trying next"
      continue
    fi

    # Free port: spawn here.
    name="openobserve-embedder-test-${p}"
    if start_obs "${p}" "${name}"; then
      SPAWNED_CONTAINER="${name}"
      SPAWNED_PORT="${p}"
      CHOSEN_PORT="${p}"
      break
    fi
  done
fi

if [ -z "${CHOSEN_PORT}" ]; then
  echo "[FAIL] Could not find or start OpenObserve on ${PORT}..$((PORT+4))"
  echo "       Check 'docker ps' for conflicting containers and free ports."
  exit 1
fi

BASE="http://localhost:${CHOSEN_PORT}"

# ---------------------------------------------------------------- 2. health
echo "[2/5] Waiting for OpenObserve health"
for i in $(seq 1 60); do
  if probe "${CHOSEN_PORT}"; then
    echo "       Healthy"
    break
  fi
  if [ "$i" -eq 60 ]; then
    echo "[FAIL] OpenObserve did not become healthy on ${CHOSEN_PORT}"
    if [ -n "${SPAWNED_CONTAINER}" ]; then
      docker logs --tail 100 "${SPAWNED_CONTAINER}"
    fi
    exit 2
  fi
  sleep 1
done

# ---------------------------------------------------------------- 3. ingest
echo "[3/5] Sending a synthetic OTLP/HTTP trace"
NOW_NS=$(python3 -c "import time; print(int(time.time() * 1_000_000_000))")
NOW_NS_END=$(python3 -c "import time; print(int(time.time() * 1_000_000_000) + 1_000_000_000)")
TRACE_ID=$(openssl rand -hex 16)
SPAN_ID=$(openssl rand -hex 8)

PAYLOAD=$(cat <<JSON
{
  "resourceSpans": [{
    "resource": {
      "attributes": [{
        "key": "service.name",
        "value": { "stringValue": "embedder-test" }
      }]
    },
    "scopeSpans": [{
      "scope": { "name": "test" },
      "spans": [{
        "traceId": "${TRACE_ID}",
        "spanId": "${SPAN_ID}",
        "name": "test.span",
        "kind": 1,
        "startTimeUnixNano": "${NOW_NS}",
        "endTimeUnixNano": "${NOW_NS_END}",
        "status": { "code": 1 }
      }]
    }]
  }]
}
JSON
)

HTTP_CODE=$(curl -s -o /tmp/obs_ingest_body -w "%{http_code}" \
  -X POST "${BASE}/api/default/v1/traces" \
  -H "Content-Type: application/json" \
  -H "stream-name: default" \
  -u "${AUTH}" \
  -d "${PAYLOAD}")

echo "       ingestion HTTP status: ${HTTP_CODE}"
if [ "${HTTP_CODE}" != "200" ]; then
  echo "[FAIL] ingestion failed — response:"
  cat /tmp/obs_ingest_body
  echo
  [ -n "${SPAWNED_CONTAINER}" ] && docker logs --tail 100 "${SPAWNED_CONTAINER}"
  exit 3
fi

# ---------------------------------------------------------------- 4. verify
echo "[4/5] Polling for the trace in the 'default' stream"
START_US=$(python3 -c "import time; print(int(time.time() * 1_000_000) - 60 * 1_000_000)")
END_US=$(python3 -c "import time; print(int(time.time() * 1_000_000) + 60 * 1_000_000)")

FOUND=0
for i in $(seq 1 20); do
  RESP=$(curl -fsS -u "${AUTH}" \
    -X POST "${BASE}/api/default/_search?type=traces" \
    -H "Content-Type: application/json" \
    -d "{
      \"query\": {
        \"sql\": \"SELECT * FROM \\\"default\\\" ORDER BY _timestamp DESC LIMIT 10\",
        \"start_time\": ${START_US},
        \"end_time\": ${END_US},
        \"from\": 0,
        \"size\": 10
      }
    }" 2>/dev/null || echo '{}')

  HITS=$(printf '%s' "${RESP}" | python3 -c "
import json, sys
try:
    r = json.load(sys.stdin)
    print(len(r.get('hits', [])))
except Exception:
    print(0)
")

  if [ "${HITS}" -gt 0 ]; then
    FOUND=1
    echo "       [OK] ${HITS} trace(s) visible in OpenObserve"
    echo "       trace_id: ${TRACE_ID}"
    break
  fi
  sleep 1
done

if [ "${FOUND}" -ne 1 ]; then
  echo "[FAIL] trace not found after 20 seconds"
  echo "       Last response from search API:"
  echo "${RESP}" | python3 -m json.tool 2>/dev/null || echo "${RESP}"
  echo
  [ -n "${SPAWNED_CONTAINER}" ] && docker logs --tail 50 "${SPAWNED_CONTAINER}"
  exit 4
fi

# ---------------------------------------------------------------- 5. streams
echo "[5/5] Stream inventory"
curl -fsS -u "${AUTH}" "${BASE}/api/default/streams" \
  | python3 -c "
import json, sys
try:
    r = json.load(sys.stdin)
    lst = r.get('list', [])
    if not lst:
        print('       (no streams yet)')
    for s in lst:
        print(f\"       {s.get('name','?')} ({s.get('stream_type','unknown')})\")
except Exception as e:
    print(f'       (parse error: {e})')
"

echo
echo "[SUCCESS] OpenObserve ingestion verified."
echo "         UI:       ${BASE}"
echo "         Email:    ${EMAIL}"
echo "         Password: ${PASSWORD}"
if [ -n "${SPAWNED_CONTAINER}" ]; then
  echo "         Container: ${SPAWNED_CONTAINER} (will be removed on exit unless KEEP=1)"
else
  echo "         Reused existing OpenObserve — nothing to clean up."
fi
