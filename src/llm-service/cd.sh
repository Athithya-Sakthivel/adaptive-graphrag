#!/usr/bin/env bash
# ==============================================================================
# src/llm-service/cd.sh — LLM Service lifecycle manager
#
# Deploys the self-contained Qwen3.5 IK LLM Service to Kubernetes.
#
# Features:
#   - 1 Replica (LLMs do not horizontally scale well without complex routing).
#   - Injects the LLM_API_KEY securely via Kubernetes Secret.
#   - Supports passing the full image reference (tag or sha256 digest) in a single
#     env var: IMAGE="ghcr.io/user/repo:tag" or IMAGE="ghcr.io/user/repo@sha256:..."
#   - Automatically sets imagePullPolicy based on mutability (latest=Always).
#   - Configures non-root execution (UID 10001) with read-only root FS.
#
# Usage:
#   bash src/llm-service/cd.sh deploy   # Apply Secret, Deployment, and Service
#   bash src/llm-service/cd.sh render   # Render manifests to disk only
#   bash src/llm-service/cd.sh delete   # Remove all LLM service resources
#   bash src/llm-service/cd.sh logs     # Tail the LLM container logs
#
# Environment Overrides:
#   NAMESPACE     (default: default)
#   IMAGE         Full image URI, e.g., "ghcr.io/user/repo@sha256:1234abcd..."
#   LLM_API_KEY   (required for deploy, injected into the container)
# ==============================================================================

set -Eeuo pipefail
IFS=$'\n\t'
umask 0077

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
NAMESPACE="${NAMESPACE:-default}"
APP_NAME="${APP_NAME:-llm-service}"
LLM_API_KEY="${LLM_API_KEY:-}"

# If IMAGE is not set, default to the GHCR latest tag
IMAGE="${IMAGE:-ghcr.io/athithya-sakthivel/qwen3.5-0.8b-ik-llama:latest}"

# The LLM is a CPU-heavy singleton. We hardcode 1 replica.
REPLICAS=1
PORT=8099
SECRET_NAME="${APP_NAME}-auth"

# Determine if it's a tag or digest
if [[ "${IMAGE}" == *"@"* ]]; then
    # SHA Digest (e.g., repo@sha256:1234) -> Immutable
    IMAGE_REPO="${IMAGE%@*}"
    IMAGE_TAG="${IMAGE#*@}"
    PULL_POLICY="IfNotPresent"
elif [[ "${IMAGE##*:}" != *"/"* && "${IMAGE}" == *":"* ]]; then
    # Tag (e.g., repo:tag or localhost:5000/repo:tag) -> Mutable if latest
    IMAGE_REPO="${IMAGE%:*}"
    IMAGE_TAG="${IMAGE##*:}"
    if [[ "${IMAGE_TAG}" == "latest" ]]; then
        PULL_POLICY="Always"
    else
        PULL_POLICY="IfNotPresent"
    fi
else
    # No tag, defaults to latest
    IMAGE_REPO="${IMAGE}"
    IMAGE_TAG="latest"
    IMAGE="${IMAGE}:latest"
    PULL_POLICY="Always"
fi

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
GEN_DIR="${GEN_DIR:-${REPO_ROOT}/src/llm-service/k8s}"

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------
C_RESET='\033[0m'; C_RED='\033[0;31m'; C_GREEN='\033[0;32m'; C_YELLOW='\033[1;33m'; C_CYAN='\033[0;36m'
log_info()   { printf "${C_CYAN}[%s]${C_RESET} [INFO]  %s\n"  "$(date -u +%H:%M:%SZ)" "$*" >&2; }
log_warn()   { printf "${C_YELLOW}[%s]${C_RESET} [WARN]  %s\n" "$(date -u +%H:%M:%SZ)" "$*" >&2; }
log_error()  { printf "${C_RED}[%s]${C_RESET} [ERROR] %s\n"   "$(date -u +%H:%M:%SZ)" "$*" >&2; }
log_success(){ printf "${C_GREEN}[%s]${C_RESET} [OK]    %s\n" "$(date -u +%H:%M:%SZ)" "$*"; }
die()        { log_error "$*"; exit 1; }

# ------------------------------------------------------------------------------
# Preflight
# ------------------------------------------------------------------------------
preflight() {
  command -v kubectl >/dev/null 2>&1 || die "kubectl not found"
  kubectl cluster-info >/dev/null 2>&1 || die "kubectl cannot reach the cluster"

  log_info "Target namespace: ${NAMESPACE}"
  log_info "Target image:     ${IMAGE}"
  log_info "Image repo:       ${IMAGE_REPO}"
  log_info "Image tag/digest: ${IMAGE_TAG}"
  log_info "Pull policy:      ${PULL_POLICY}"
  log_info "Replicas:         ${REPLICAS}"
}

# ------------------------------------------------------------------------------
# Manifest Generation
# ------------------------------------------------------------------------------
render_manifests() {
  mkdir -p "${GEN_DIR}"
  log_info "Rendering manifests to ${GEN_DIR}"

  # ---------------------------------------------------------------------------
  # Deployment
  # ---------------------------------------------------------------------------
  cat > "${GEN_DIR}/deployment.yaml" <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ${APP_NAME}
  namespace: ${NAMESPACE}
  labels:
    app.kubernetes.io/name: ${APP_NAME}
spec:
  replicas: ${REPLICAS}
  strategy:
    type: Recreate # Important: Do not run old and new LLM simultaneously (OOM risk)
  selector:
    matchLabels:
      app.kubernetes.io/name: ${APP_NAME}
  template:
    metadata:
      labels:
        app.kubernetes.io/name: ${APP_NAME}
    spec:
      enableServiceLinks: false
      securityContext:
        runAsNonRoot: true
        runAsUser: 10001
        runAsGroup: 10001
        fsGroup: 10001
        seccompProfile:
          type: RuntimeDefault
      containers:
        - name: ${APP_NAME}
          image: ${IMAGE}
          imagePullPolicy: ${PULL_POLICY}
          env:
            - name: LLM_API_KEY
              valueFrom:
                secretKeyRef:
                  name: ${SECRET_NAME}
                  key: LLM_API_KEY
          ports:
            - name: http
              containerPort: ${PORT}
              protocol: TCP
          resources:
            requests:
              cpu: 500m
              memory: 1Gi
            limits:
              cpu: "4"
              memory: 4Gi
          securityContext:
            allowPrivilegeEscalation: false
            readOnlyRootFilesystem: true
            capabilities: { drop: ["ALL"] }
          volumeMounts:
            - name: tmp
              mountPath: /tmp
          startupProbe:
            httpGet: { path: /health, port: http }
            periodSeconds: 10
            failureThreshold: 30
          readinessProbe:
            httpGet: { path: /health, port: http }
            periodSeconds: 5
            failureThreshold: 3
          livenessProbe:
            httpGet: { path: /health, port: http }
            periodSeconds: 15
            failureThreshold: 3
      volumes:
        - name: tmp
          emptyDir: {}
EOF

  # ---------------------------------------------------------------------------
  # Service
  # ---------------------------------------------------------------------------
  cat > "${GEN_DIR}/service.yaml" <<EOF
apiVersion: v1
kind: Service
metadata:
  name: ${APP_NAME}
  namespace: ${NAMESPACE}
  labels:
    app.kubernetes.io/name: ${APP_NAME}
spec:
  type: ClusterIP
  selector:
    app.kubernetes.io/name: ${APP_NAME}
  ports:
    - name: http
      port: ${PORT}
      targetPort: http
      protocol: TCP
EOF
}

# ------------------------------------------------------------------------------
# Commands
# ------------------------------------------------------------------------------
cmd_render() {
  preflight
  render_manifests
  log_success "Manifests rendered to ${GEN_DIR}"
}

cmd_deploy() {
  preflight

  if [[ -z "${LLM_API_KEY}" ]]; then
    die "LLM_API_KEY is not set. Export it before running deploy: export LLM_API_KEY='your-key'"
  fi

  log_info "Creating/updating API key secret..."
  kubectl create secret generic "${SECRET_NAME}" \
    --namespace "${NAMESPACE}" \
    --from-literal=LLM_API_KEY="${LLM_API_KEY}" \
    --dry-run=client -o yaml | kubectl apply -f - >/dev/null

  render_manifests

  log_info "Applying Service..."
  kubectl apply -f "${GEN_DIR}/service.yaml" --server-side --force-conflicts

  log_info "Applying Deployment (1 replica)..."
  kubectl apply -f "${GEN_DIR}/deployment.yaml" --server-side --force-conflicts

  log_info "Waiting for LLM model to load (timeout=300s)..."
  if ! kubectl rollout status "deployment/${APP_NAME}" -n "${NAMESPACE}" --timeout=300s; then
    log_error "Deployment did not become ready. Logs:"
    kubectl -n "${NAMESPACE}" logs -l "app.kubernetes.io/name=${APP_NAME}" --tail=50 || true
    die "Deployment failed"
  fi

  log_success "LLM Service deployed successfully"
}

cmd_delete() {
  preflight
  log_warn "Deleting LLM Service resources..."
  kubectl -n "${NAMESPACE}" delete deployment "${APP_NAME}" --ignore-not-found >/dev/null 2>&1 || true
  kubectl -n "${NAMESPACE}" delete svc "${APP_NAME}" --ignore-not-found >/dev/null 2>&1 || true
  kubectl -n "${NAMESPACE}" delete secret "${SECRET_NAME}" --ignore-not-found >/dev/null 2>&1 || true
  log_success "LLM Service resources removed."
}

cmd_logs() {
  preflight
  kubectl -n "${NAMESPACE}" logs -l "app.kubernetes.io/name=${APP_NAME}" -c "${APP_NAME}" --tail=100 -f
}

usage() {
  cat <<EOF
Usage: $(basename "$0") <command>

Commands:
  deploy    Apply Secret, Deployment (1 replica), and Service
  render    Render manifests to ${GEN_DIR}
  delete    Remove all LLM Service resources
  logs      Tail the LLM container logs
  help      Show this help

Environment:
  NAMESPACE     (default: default)
  IMAGE         Full image URI (tag or sha256 digest).
                Example: "ghcr.io/user/repo:latest"
                         "ghcr.io/user/repo@sha256:1234abcd..."
                (default: ghcr.io/athithya-sakthivel/qwen3.5-0.8b-ik-llama:latest)
  LLM_API_KEY   (required for deploy; used to authenticate API requests)

Example:
  export LLM_API_KEY="super-secret-llm-key"

  # Deploy latest
  IMAGE="ghcr.io/athithya-sakthivel/qwen3.5-0.8b-ik-llama:latest" ./src/llm-service/cd.sh deploy

  # Deploy immutable sha
  IMAGE="ghcr.io/athithya-sakthivel/qwen3.5-0.8b-ik-llama@sha256:abcd1234..." ./src/llm-service/cd.sh deploy
EOF
}

main() {
  if (( $# == 0 )); then usage; exit 2; fi
  local cmd="$1"; shift || true

  case "${cmd}" in
    deploy)   cmd_deploy ;;
    render)   cmd_render ;;
    delete)   cmd_delete ;;
    logs)     cmd_logs ;;
    help|--help|-h) usage ;;
    *) usage; exit 2 ;;
  esac
}

main "$@"
