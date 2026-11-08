#!/usr/bin/env bash
# setup.sh — Bring up OpenObserve for local dense-service testing.
#
# Idempotent:
#   - creates the shared docker network if missing
#   - removes and recreates the openobserve container
#   - persists data under ./data
#
# Usage:
#   ./setup.sh              # start
#   ./setup.sh teardown     # stop + remove container + network
#
# Env overrides:
#   NETWORK   (default: dense-net)
#   NAME      (default: openobserve)
#   PORT      (default: 5080)
#   EMAIL     (default: root@example.com)
#   PASSWORD  (default: Complexpass#123)

set -euo pipefail
cd "$(dirname "$0")"

NETWORK="${NETWORK:-dense-net}"
NAME="${NAME:-openobserve}"
PORT="${PORT:-5080}"
EMAIL="${EMAIL:-root@example.com}"
PASSWORD="${PASSWORD:-Complexpass#123}"
DATA_DIR="${DATA_DIR:-$PWD/data}"

teardown() {
  echo "[setup] Removing container ${NAME}"
  docker rm -f "${NAME}" >/dev/null 2>&1 || true
  echo "[setup] Removing network ${NETWORK}"
  docker network rm "${NETWORK}" >/dev/null 2>&1 || true
}

case "${1:-}" in
  teardown|down|stop)
    teardown
    exit 0
    ;;
  ""|up|start)
    ;;
  *)
    echo "Usage: $0 [up|teardown]" >&2
    exit 2
    ;;
esac

# Create the network (shared with the dense-service).
if ! docker network inspect "${NETWORK}" >/dev/null 2>&1; then
  echo "[setup] Creating network ${NETWORK}"
  docker network create "${NETWORK}"
else
  echo "[setup] Network ${NETWORK} already exists"
fi

# Recreate the container.
if docker ps -aq -f "name=^/${NAME}$" | grep -q .; then
  echo "[setup] Removing existing container ${NAME}"
  docker rm -f "${NAME}" >/dev/null 2>&1
fi

mkdir -p "${DATA_DIR}"

echo "[setup] Starting OpenObserve (port ${PORT}, data ${DATA_DIR})"
docker run -d \
  --name "${NAME}" \
  --network "${NETWORK}" \
  -p "${PORT}:5080" \
  -v "${DATA_DIR}:/data" \
  -e ZO_DATA_DIR=/data \
  -e ZO_ROOT_USER_EMAIL="${EMAIL}" \
  -e ZO_ROOT_USER_PASSWORD="${PASSWORD}" \
  -e ZO_INGEST_ALLOWED_UPTO=720 \
  -e ZO_COMPACT_DATA_RETENTION_DAYS=30 \
  public.ecr.aws/zinclabs/openobserve:latest >/dev/null

echo "[setup] Waiting for OpenObserve to be reachable"
for _ in $(seq 1 60); do
  if curl -fsS --max-time 2 -u "${EMAIL}:${PASSWORD}" \
       "http://127.0.0.1:${PORT}/healthz" >/dev/null 2>&1; then
    echo "[setup] OpenObserve ready."
    echo "[setup]   UI:       http://localhost:${PORT}"
    echo "[setup]   Email:    ${EMAIL}"
    echo "[setup]   Password: ${PASSWORD}"
    echo "[setup]   Auth b64: $(printf '%s' "${EMAIL}:${PASSWORD}" | base64 | tr -d '\n')"
    exit 0
  fi
  sleep 1
done

echo "[setup] ERROR: OpenObserve did not become reachable — logs:"
docker logs --tail 100 "${NAME}"
exit 1
