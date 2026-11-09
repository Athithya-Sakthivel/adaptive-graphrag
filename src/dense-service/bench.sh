#!/usr/bin/env bash
# bench.sh — A/B sweep of DENSE_PARALLEL on a single image.
#
# Prerequisites:
#   - the image at $IMAGE has the threads=1 fix baked in
#   - nothing else is bound to 8200-8299
#
# Usage:
#   bash src/dense-service/bench.sh
#   IMAGE=ghcr.io/.../dense-embedder:2026-11-08-... bash src/dense-service/bench.sh
#   CONFIGS="7 8 9 10 12" bash src/dense-service/bench.sh
#
# Metrics reported per config:
#   - 5 timed runs (runs 1-2 warmup, 3-5 measured); median of 3-5
#   - /proc/1/status Threads:  (should be ~20-30 with threads=1)
#   - peak RSS via docker stats
#   - mpstat %usr/%sys/%idle during the measured window

set -Eeuo pipefail

IMAGE="${IMAGE:-ghcr.io/athithya-sakthivel/dense-embedder:latest}"
CONFIGS="${CONFIGS:-7 8 9 10}"
BASE_PORT=8200
N_TEXTS=64
TOKENS=512
N_WARMUP=2
N_MEASURE=3
TIMEOUT_READY=180

# ---- helpers ----
c_g=$'\033[0;32m'; c_y=$'\033[1;33m'; c_b=$'\033[0;34m'; c_r=$'\033[0m'
log()  { printf '%s==>%s %s\n' "$c_b" "$c_r" "$*" >&2; }
pass() { printf '%s[OK]%s %s\n' "$c_g" "$c_r" "$*" >&2; }
warn() { printf '%s[WARN]%s %s\n' "$c_y" "$c_r" "$*" >&2; }
fail() { printf '%s[FAIL]%s %s\n' "$c_r" "$c_r" "$*" >&2; exit 1; }

make_payload() {
    python3 -c "import json; print(json.dumps({'texts': ['word ' * ${TOKENS}] * ${N_TEXTS}}))"
}

wait_ready() {
    local port="$1" deadline=$(( $(date +%s) + TIMEOUT_READY ))
    while [ "$(date +%s)" -lt "$deadline" ]; do
        if curl -fsS --max-time 2 "http://127.0.0.1:${port}/readyz" >/dev/null 2>&1; then
            return 0
        fi
        sleep 2
    done
    return 1
}

# ---- preflight ----
command -v docker  >/dev/null || fail "docker not found"
command -v curl    >/dev/null || fail "curl not found"
command -v python3 >/dev/null || fail "python3 not found"
docker info >/dev/null 2>&1 || fail "docker daemon unavailable"
[[ -n "$(docker image inspect "$IMAGE" 2>/dev/null)" ]] || fail "image not found: $IMAGE"

printf 'Image:    %s\n' "$IMAGE"
printf 'Configs:  %s\n' "$CONFIGS"
printf 'Workload: %d texts x %d tokens = %d tokens/request\n' \
    "$N_TEXTS" "$TOKENS" "$((N_TEXTS * TOKENS))"
printf 'Runs:     %d warmup + %d measured per config\n\n' "$N_WARMUP" "$N_MEASURE"

# ---- verify the image has the threads=1 fix ----
# This is the single most important precheck. If the image still ships
# the old code, every config will be equally slow and the sweep is meaningless.
log "verifying threads=1 fix is present in the image"
INSPECT=$(docker run --rm --entrypoint sh "$IMAGE" -c '
python3 - <<PY
import inspect
try:
    from fastembed import TextEmbedding
    sig = inspect.signature(TextEmbedding.__init__)
    print("fastembed_accepts_threads=", "threads" in sig.parameters, sep="")
except Exception as e:
    print("fastembed_error=", e, sep="")
PY
')
echo "  $INSPECT"
echo "$INSPECT" | grep -q "fastembed_accepts_threads=True" \
    || fail "installed fastembed does not accept 'threads=' — bump fastembed in requirements.txt"

# Grep the baked host_dense.py for 'threads=' on the TextEmbedding call.
docker run --rm --entrypoint sh "$IMAGE" -c \
    'grep -q "threads=1" /app/host_dense.py && echo ok' >/dev/null \
    || fail "baked host_dense.py has no 'threads=1' — rebuild the image with the fix"
pass "image is the fixed one"

# ---- results table ----
declare -A MEDIANS THREADS RSS
RESULTS=()
PORT=$BASE_PORT

for p in $CONFIGS; do
    PORT=$((PORT + 1))
    NAME="dense-bench-${p}"
    docker rm -f "$NAME" >/dev/null 2>&1 || true

    log "starting $NAME on :${PORT} (DENSE_PARALLEL=${p})"
    docker run -d --name "$NAME" \
        -p "${PORT}:8200" \
        -e "DENSE_PARALLEL=${p}" \
        "$IMAGE" >/dev/null

    if ! wait_ready "$PORT"; then
        warn "config p=${p}: not ready after ${TIMEOUT_READY}s; logs:"
        docker logs --tail 40 "$NAME" 2>&1 | sed 's/^/    /' >&2
        docker rm -f "$NAME" >/dev/null 2>&1 || true
        continue
    fi

    THREADS["$p"]=$(docker exec "$NAME" sh -c 'grep Threads /proc/1/status' | awk '{print $2}')
    log "  threads in PID 1: ${THREADS[$p]}"

    PAYLOAD_FILE=$(mktemp)
    make_payload > "$PAYLOAD_FILE"

    # start mpstat sampling in the background
    MPSTAT_LOG=$(mktemp)
    if command -v mpstat >/dev/null 2>&1; then
        mpstat 1 $(( N_WARMUP + N_MEASURE + 5 )) > "$MPSTAT_LOG" 2>&1 &
        MPSTAT_PID=$!
    else
        MPSTAT_PID=""
        warn "mpstat not installed; skipping CPU sampling (apt install sysstat)"
    fi

    TIMES=()
    for i in $(seq 1 $((N_WARMUP + N_MEASURE))); do
        T=$( { /usr/bin/time -f "%e" curl -fsS -o /dev/null \
                -X POST "http://127.0.0.1:${PORT}/embed" \
                -H 'Content-Type: application/json' \
                -d "@${PAYLOAD_FILE}" ; } 2>&1 )
        TIMES+=("$T")
        if [ "$i" -le "$N_WARMUP" ]; then
            printf '  run %d/%d (warmup) %ss\n' "$i" "$((N_WARMUP+N_MEASURE))" "$T"
        else
            printf '  run %d/%d (measure) %ss\n' "$i" "$((N_WARMUP+N_MEASURE))" "$T"
        fi
    done

    # median of measured runs (skip first N_WARMUP entries)
    MEDIAN=$(printf '%s\n' "${TIMES[@]:$N_WARMUP}" | sort -n | awk '
        {a[NR]=$1}
        END { if (NR==0) print "nan";
              else if (NR%2==1) print a[(NR+1)/2];
              else print (a[NR/2]+a[NR/2+1])/2 }')

    MEDIANS["$p"]="$MEDIAN"

    # peak RSS
    RSS["$p"]=$(docker stats --no-stream --format '{{.MemUsage}}' "$NAME" 2>/dev/null | awk '{print $1}')

    if [ -n "$MPSTAT_PID" ]; then
        wait "$MPSTAT_PID" 2>/dev/null || true
        echo "  mpstat summary (all cores, last 3 samples):"
        tail -n +5 "$MPSTAT_LOG" | tail -3 | awk '{printf "    %%usr=%s  %%sys=%s  %%idle=%s\n", $4, $5, $NF}'
    fi

    docker rm -f "$NAME" >/dev/null 2>&1 || true
    rm -f "$PAYLOAD_FILE" "$MPSTAT_LOG"

    RESULTS+=("$p")
done

# ---- comparison table ----
echo
echo "========================================="
echo "  DENSE_PARALLEL sweep — ${N_TEXTS}x${TOKENS} tokens"
echo "========================================="
printf '  %-10s %-12s %-12s %-12s %-12s\n' \
    "parallel" "median(s)" "texts/s" "tokens/s" "threads"
echo "  ---------------------------------------------------------------------"

BASELINE=""
for p in "${RESULTS[@]}"; do
    m="${MEDIANS[$p]:-nan}"
    if [ "$m" = "nan" ]; then
        printf '  %-10s %-12s %-12s %-12s %-12s\n' "$p" "failed" "-" "-" "${THREADS[$p]:-?}"
        continue
    fi
    tps=$(awk -v n="$N_TEXTS" -v t="$m" 'BEGIN{printf "%.1f", n/t}')
    tokps=$(awk -v n="$((N_TEXTS*TOKENS))" -v t="$m" 'BEGIN{printf "%.0f", n/t}')
    [ -z "$BASELINE" ] && BASELINE="$m"
    delta=$(awk -v a="$BASELINE" -v b="$m" 'BEGIN{printf "%+.1f%%", (b-a)/a*100}')
    printf '  %-10s %-12s %-12s %-12s %-12s  (%s vs first)\n' \
        "$p" "$m" "$tps" "$tokps" "${THREADS[$p]:-?}" "$delta"
done

echo "========================================="
echo
echo "Interpretation:"
echo "  - threads should be ~20-30 for every config with threads=1"
echo "  - if threads > 100, the fix is not applied to this image"
echo "  - lowest median wins; differences <5% are noise"
