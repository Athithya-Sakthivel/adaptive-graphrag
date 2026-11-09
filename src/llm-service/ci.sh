#!/usr/bin/env bash
# =============================================================================
# src/llm-service/ci.sh
#
# Purpose:
#   Builds the self-contained IK LLM Docker image, verifies it works correctly
#   via HTTP smoke tests, and pushes it to GitHub Container Registry (GHCR).
#
# Features:
#   - Uses Buildx with host CA secrets to securely compile inside the container.
#   - Bakes the GGUF model directly into the image for zero-dependency runtime.
#   - Runs ephemeral smoke tests against /health, /tokenize, and /completion.
#   - Passes optional API keys to the container during testing.
#
# Usage (from anywhere, but typically from repo root):
#   cd /workspace
#   export GIT_PAT="ghp_..."       # Required: GitHub PAT with write:packages permission
#   export LLM_API_KEY="super-secret-llm-key"  # Optional: API key for the LLM server (if your Dockerfile CMD uses --api-key)
#
#   bash src/llm-service/ci.sh
# =============================================================================

set -Eeuo pipefail
IFS=$'\n\t'

if (( $# != 0 )); then
    printf 'ERROR: ci.sh accepts no positional arguments.\n' >&2
    exit 2
fi

C_RESET=$'\033[0m'
C_BLUE=$'\033[0;34m'
C_GREEN=$'\033[0;32m'
C_YELLOW=$'\033[1;33m'
C_RED=$'\033[0;31m'

log()  { printf '%s==>%s %s\n' "$C_BLUE" "$C_RESET" "$*" >&2; }
pass() { printf '%s[OK]%s %s\n' "$C_GREEN" "$C_RESET" "$*" >&2; }
warn() { printf '%s[WARN]%s %s\n' "$C_YELLOW" "$C_RESET" "$*" >&2; }
fail() { printf '%sERROR:%s %s\n' "$C_RED" "$C_RESET" "$*" >&2; exit 1; }
step() { printf '\n%s==>%s [%s]\n' "$C_BLUE" "$C_RESET" "$1" >&2; }

SERVICE_DIR="$(
    cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
    pwd
)"

REPO_ROOT="$(
    git -C "$SERVICE_DIR" rev-parse --show-toplevel 2>/dev/null
)" || fail "Cannot discover Git repository root."

DOCKERFILE="${SERVICE_DIR}/Dockerfile"
MODEL_REL_PATH="models/llm/Qwen3.5-0.8B-Q4_K_M.gguf"
MODEL_PATH="${REPO_ROOT}/${MODEL_REL_PATH}"

UBUNTU_VERSION="${UBUNTU_VERSION:-26.04}"
IK_LLAMA_REF="${IK_LLAMA_REF:-39fc3128ccc428314a64b98de09af24bd7bb4990}"
BUILD_JOBS="${BUILD_JOBS:-1}"
DOCKER_PLATFORM="${DOCKER_PLATFORM:-linux/amd64}"
GHCR_REGISTRY="${GHCR_REGISTRY:-ghcr.io}"
GHCR_USER="${GHCR_USER:-athithya-sakthivel}"
IMAGE_NAME="${IMAGE_NAME:-qwen3.5-0.8b-ik-llama}"
TEST_START_TIMEOUT="${TEST_START_TIMEOUT:-300}"
TEST_COMPLETION_TIMEOUT="${TEST_COMPLETION_TIMEOUT:-180}"

# Optional API Key (evaluates to empty string if not exported)
LLM_API_KEY="${LLM_API_KEY:-}"

GHCR_USER="${GHCR_USER,,}"
IMAGE_REPO="${GHCR_REGISTRY%/}/${GHCR_USER}/${IMAGE_NAME}"

GIT_SHA="$(
    git -C "$REPO_ROOT" rev-parse --short=12 HEAD
)"

DIRTY_SUFFIX=""
if [[ -n "$(
    git -C "$REPO_ROOT" status --porcelain
)" ]]; then
    DIRTY_SUFFIX="-amd-cpu"
fi

BUILD_TIMESTAMP="$(
    TZ=Asia/Kolkata date +%Y-%m-%d-%H-%M-%S
)"

TAG="${BUILD_TIMESTAMP}--${GIT_SHA}${DIRTY_SUFFIX}"
IMAGE_TAGGED="${IMAGE_REPO}:${TAG}"
IMAGE_LATEST="${IMAGE_REPO}:latest"
TEST_CONTAINER="llm-service-ci-${BASHPID:-$$}"

cleanup() {
    local rc=$?
    trap - EXIT INT TERM

    if (( rc != 0 )); then
        if docker container inspect "$TEST_CONTAINER" >/dev/null 2>&1; then
            warn "Smoke-test container logs:"
            docker logs --tail 160 "$TEST_CONTAINER" 2>&1 \
                | sed 's/^/    /' >&2 || true
        fi
    fi

    docker rm -f "$TEST_CONTAINER" >/dev/null 2>&1 || true
    exit "$rc"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# -----------------------------------------------------------------------------
# CA Bundle Preflight
# -----------------------------------------------------------------------------
HOST_CA_BUNDLE="${SSL_CERT_FILE:-/etc/ssl/certs/ca-certificates.crt}"

[[ -s "$HOST_CA_BUNDLE" ]] \
    || fail "Host CA bundle missing: $HOST_CA_BUNDLE"

log "Verifying host CA bundle against Ubuntu archive"

curl \
    --cacert "$HOST_CA_BUNDLE" \
    --fail \
    --silent \
    --show-error \
    --max-time 30 \
    "https://archive.ubuntu.com/ubuntu/dists/resolute/InRelease" \
    -o /dev/null \
    || fail "Host CA bundle cannot validate Ubuntu HTTPS archive."

pass "Host CA bundle verified."

# -----------------------------------------------------------------------------
# Preflight
# -----------------------------------------------------------------------------
step preflight

[[ -f "$DOCKERFILE" ]] \
    || fail "Dockerfile missing: ${DOCKERFILE}"

[[ -s "$MODEL_PATH" ]] \
    || fail "Model missing or empty: ${MODEL_PATH}"

[[ -n "${GIT_PAT:-}" ]] \
    || fail "GIT_PAT is required. Provide a GitHub PAT (classic) with write:packages."

[[ "$BUILD_JOBS" =~ ^[1-9][0-9]*$ ]] \
    || fail "BUILD_JOBS must be a positive integer."

[[ "$IK_LLAMA_REF" =~ ^[0-9a-f]{40}$ ]] \
    || fail "IK_LLAMA_REF must be a full 40-character Git commit SHA."

[[ "$TEST_START_TIMEOUT" =~ ^[1-9][0-9]*$ ]] \
    || fail "TEST_START_TIMEOUT must be a positive integer."

[[ "$TEST_COMPLETION_TIMEOUT" =~ ^[1-9][0-9]*$ ]] \
    || fail "TEST_COMPLETION_TIMEOUT must be a positive integer."

[[ "$GHCR_USER" =~ ^[a-z0-9]([a-z0-9-]{0,37}[a-z0-9])?$ ]] \
    || fail "Invalid GHCR_USER."

[[ "$IMAGE_NAME" =~ ^[a-z0-9]+([._-][a-z0-9]+)*$ ]] \
    || fail "IMAGE_NAME must be lowercase and registry-safe."

for tool in docker git curl python3 sha256sum; do
    command -v "$tool" >/dev/null 2>&1 \
        || fail "Required command not found: ${tool}"
done

docker info >/dev/null 2>&1 \
    || fail "Docker daemon unavailable."

docker buildx version >/dev/null 2>&1 \
    || fail "Docker Buildx is required. Install docker-buildx-plugin."

docker buildx inspect --bootstrap >/dev/null 2>&1 \
    || fail "Docker Buildx builder could not start."

HOST_ARCH="$(uname -m)"
case "${DOCKER_PLATFORM}:${HOST_ARCH}" in
    linux/amd64:x86_64) ;;
    linux/arm64:aarch64) ;;
    *)
        fail "Native build platform does not match host architecture: ${HOST_ARCH}"
        ;;
esac

MODEL_SHA256="$(
    sha256sum "$MODEL_PATH" | awk '{print $1}'
)"

printf 'Service directory: %s\n' "$SERVICE_DIR"
printf 'Repository root:   %s\n' "$REPO_ROOT"
printf 'Model:             %s\n' "$MODEL_PATH"
printf 'Model SHA256:      %s\n' "$MODEL_SHA256"
printf 'Upstream IK ref:   %s\n' "$IK_LLAMA_REF"
printf 'Ubuntu base:       %s\n' "$UBUNTU_VERSION"
printf 'Native build:      ON\n'
printf 'Runtime config:    ctx=4096 t=4 tb=7 b=2048 ub=1024 np=1\n'
printf 'Docker platform:   %s\n' "$DOCKER_PLATFORM"
printf 'Git tag:           %s\n' "$TAG"
printf 'Image tagged:      %s\n' "$IMAGE_TAGGED"
printf 'Image latest:      %s\n' "$IMAGE_LATEST"
printf 'Build context:     %s\n' "$REPO_ROOT"
printf 'Model packaging:   baked into image\n'
printf 'Build policy:      --pull --no-cache\n'
if [[ -n "$LLM_API_KEY" ]]; then
    printf 'LLM API Key:       provided (smoke tests will use Bearer auth)\n'
else
    printf 'LLM API Key:       not set (smoke tests will run without auth)\n'
fi

pass "Preflight passed."

# -----------------------------------------------------------------------------
# Build
# -----------------------------------------------------------------------------
step build

log "Building Ubuntu ${UBUNTU_VERSION} image."

docker buildx build \
    --pull \
    --no-cache \
    --load \
    --platform "$DOCKER_PLATFORM" \
    --progress plain \
    --secret "id=host_ca,src=${HOST_CA_BUNDLE}" \
    --file "$DOCKERFILE" \
    --build-arg "IK_LLAMA_REF=${IK_LLAMA_REF}" \
    --build-arg "BUILD_JOBS=${BUILD_JOBS}" \
    --tag "$IMAGE_TAGGED" \
    --tag "$IMAGE_LATEST" \
    "$REPO_ROOT" \
    || fail "Docker Buildx build failed."

pass "Built ${IMAGE_TAGGED}"

# -----------------------------------------------------------------------------
# Verify image exists
# -----------------------------------------------------------------------------
step image-validation

docker image inspect "$IMAGE_TAGGED" >/dev/null 2>&1 \
    || fail "Tagged image not found locally."

IMAGE_SIZE="$(
    docker image inspect \
        --format '{{.Size}}' \
        "$IMAGE_TAGGED"
)"

printf 'Image size: %s bytes\n' "$IMAGE_SIZE"
pass "Image inspection passed."

# -----------------------------------------------------------------------------
# Start test container
# -----------------------------------------------------------------------------
step container-start

docker rm -f "$TEST_CONTAINER" >/dev/null 2>&1 || true

# Pass the API key into the container at runtime so the CMD can use it
DOCKER_RUN_ENV=()
if [[ -n "$LLM_API_KEY" ]]; then
    DOCKER_RUN_ENV+=(-e "LLM_API_KEY=${LLM_API_KEY}")
fi

docker run -d \
    --name "$TEST_CONTAINER" \
    --init \
    --publish '127.0.0.1::8099' \
    --stop-timeout 45 \
    "${DOCKER_RUN_ENV[@]}" \
    "$IMAGE_TAGGED" >/dev/null \
    || fail "Could not start the test container."

PORT_MAPPING="$(
    docker port "$TEST_CONTAINER" 8099/tcp 2>/dev/null \
        | head -n 1 || true
)"

TEST_HOST_PORT="${PORT_MAPPING##*:}"

[[ "$TEST_HOST_PORT" =~ ^[0-9]+$ ]] \
    || fail "Could not determine allocated Docker port: ${PORT_MAPPING}"

BASE_URL="http://127.0.0.1:${TEST_HOST_PORT}"
log "Temporary endpoint: ${BASE_URL}"

# Set up curl auth headers if the key is present
CURL_AUTH=()
if [[ -n "$LLM_API_KEY" ]]; then
    CURL_AUTH+=(-H "Authorization: Bearer ${LLM_API_KEY}")
fi

# -----------------------------------------------------------------------------
# Wait for health
# -----------------------------------------------------------------------------
step health

READY=0
for ((i=1; i<=TEST_START_TIMEOUT; i++)); do
    RUNNING="$(
        docker inspect \
            --format '{{.State.Running}}' \
            "$TEST_CONTAINER" 2>/dev/null || echo false
    )"

    [[ "$RUNNING" == true ]] \
        || fail "LLM container exited before becoming healthy."

    if curl -fsS --max-time 2 "${CURL_AUTH[@]}" \
        "${BASE_URL}/health" >/dev/null 2>&1; then
        READY=1
        break
    fi
    sleep 1
done

(( READY == 1 )) \
    || fail "LLM server did not become healthy within ${TEST_START_TIMEOUT}s."

HEALTH_JSON="$(
    curl -fsS --max-time 5 "${CURL_AUTH[@]}" "${BASE_URL}/health"
)" || fail "Health endpoint stopped responding."

printf '%s' "$HEALTH_JSON" | python3 -c '
import json, sys

obj = json.load(sys.stdin)

assert isinstance(obj, dict), "health response must be a JSON object"

if "status" in obj:
    assert str(obj["status"]).lower() in ("ok", "healthy"), \
        f"unexpected health status: {obj[\"status\"]}"

print("   health response:", obj)
' || fail "Invalid health response."

pass "Health check passed."

# -----------------------------------------------------------------------------
# Tokenizer test
# -----------------------------------------------------------------------------
step tokenize

TOKENIZE_BODY="$(
    python3 -c '
import json
print(json.dumps({
    "content": "hello world",
    "add_special": True,
    "parse_special": True
}))
'
)"

TOKENIZE_JSON="$(
    curl -fsS \
        --max-time 30 \
        "${CURL_AUTH[@]}" \
        -X POST "${BASE_URL}/tokenize" \
        -H 'Content-Type: application/json' \
        -d "$TOKENIZE_BODY"
)" || fail "POST /tokenize failed."

printf '%s' "$TOKENIZE_JSON" | python3 -c '
import json, sys

obj = json.load(sys.stdin)
tokens = obj.get("tokens")

assert isinstance(tokens, list) and len(tokens) > 0, \
    f"invalid tokenizer response: {obj}"

assert all(isinstance(t, int) for t in tokens), \
    "token IDs must be integers"

print("   token count:", len(tokens))
' || fail "Tokenizer response validation failed."

pass "Tokenizer test passed."

# -----------------------------------------------------------------------------
# Real inference test
# -----------------------------------------------------------------------------
step completion

COMPLETION_BODY="$(
    python3 -c '
import json
print(json.dumps({
    "prompt": "<|im_start|>user\nReply with the word OK.\n<|im_end|>\n<|im_start|>assistant\n",
    "n_predict": 8,
    "temperature": 0.0,
    "top_k": 1,
    "cache_prompt": False,
    "stream": False
}))
'
)"

COMPLETION_JSON="$(
    curl -fsS \
        --max-time "$TEST_COMPLETION_TIMEOUT" \
        "${CURL_AUTH[@]}" \
        -X POST "${BASE_URL}/completion" \
        -H 'Content-Type: application/json' \
        -d "$COMPLETION_BODY"
)" || fail "POST /completion failed."

printf '%s' "$COMPLETION_JSON" | python3 -c '
import json, sys

obj = json.load(sys.stdin)
content = obj.get("content")

assert isinstance(content, str) and content.strip(), \
    f"empty completion response: {obj}"

timings = obj.get("timings", {})
assert isinstance(timings, dict), "timings must be a JSON object"

print("   generated characters:", len(content))
if timings:
    print("   timing keys:", ", ".join(sorted(timings.keys())))
' || fail "Completion response validation failed."

pass "Real inference test passed."

# -----------------------------------------------------------------------------
# GHCR authentication & Push
# -----------------------------------------------------------------------------
step publish

log "Authenticating to ${GHCR_REGISTRY} as ${GHCR_USER}."

printf '%s' "$GIT_PAT" \
    | docker login "$GHCR_REGISTRY" \
        --username "$GHCR_USER" \
        --password-stdin >/dev/null \
    || fail "GHCR authentication failed. Check PAT write:packages permission."

pass "GHCR authentication passed."

log "Pushing ${IMAGE_TAGGED}"
docker push "$IMAGE_TAGGED" \
    || fail "Could not push ${IMAGE_TAGGED}."
pass "Pushed timestamped image."

log "Pushing ${IMAGE_LATEST}"
docker push "$IMAGE_LATEST" \
    || fail "Could not push ${IMAGE_LATEST}."
pass "Pushed latest image."

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
step summary

printf '\n'
printf '%s\n' '============================================'
printf '%s\n' '       BUILD / TEST / PUSH SUCCEEDED'
printf '%s\n' '============================================'
printf '  %s\n' "$IMAGE_TAGGED"
printf '  %s\n' "$IMAGE_LATEST"
printf '%s\n' '============================================'
printf '\n'
printf 'The image includes the GGUF model.\n'
printf 'No external model volume is required.\n'
printf '\n'
