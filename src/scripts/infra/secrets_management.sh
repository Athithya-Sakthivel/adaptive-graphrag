#!/usr/bin/env bash
set -euo pipefail

ARGO_NS="argocd"
ESO_NS="external-secrets-system"
RELOADER_NS="reloader"

wait_for_app_synced_healthy() {
  local app="$1"
  local timeout_seconds="${2:-900}"
  local interval_seconds="${3:-5}"

  local elapsed=0
  local sync_status=""
  local health_status=""

  echo "==> Waiting for ArgoCD application: ${app} (Synced + Healthy)"

  while true; do
    sync_status="$(kubectl get application "${app}" -n "${ARGO_NS}" -o jsonpath='{.status.sync.status}' 2>/dev/null || true)"
    health_status="$(kubectl get application "${app}" -n "${ARGO_NS}" -o jsonpath='{.status.health.status}' 2>/dev/null || true)"

    if [[ "${sync_status}" == "Synced" && "${health_status}" == "Healthy" ]]; then
      echo "==> ${app} is Synced and Healthy"
      return 0
    fi

    if [[ "${elapsed}" -ge "${timeout_seconds}" ]]; then
      echo "ERROR: timeout waiting for application ${app}"
      echo "sync=${sync_status:-<none>} health=${health_status:-<none>}"
      kubectl get application "${app}" -n "${ARGO_NS}" -o yaml || true
      return 1
    fi

    sleep "${interval_seconds}"
    elapsed=$((elapsed + interval_seconds))
  done
}

wait_for_crd_established() {
  local crd="$1"
  local timeout_seconds="${2:-300}"

  echo "==> Waiting for CRD: ${crd}"
  kubectl wait --for=condition=Established "crd/${crd}" --timeout="${timeout_seconds}s"
}

wait_for_deployment_available() {
  local ns="$1"
  local deploy="$2"
  local timeout_seconds="${3:-300}"

  echo "==> Waiting for deployment: ${ns}/${deploy}"
  kubectl wait --for=condition=available "deployment/${deploy}" -n "${ns}" --timeout="${timeout_seconds}s"
}

wait_for_clustersecretstore() {
  local name="$1"
  local timeout_seconds="${2:-600}"
  local interval_seconds="${3:-5}"

  local elapsed=0
  local ready_status=""
  local phase_status=""

  echo "==> Waiting for ClusterSecretStore: ${name} to become Ready"

  while true; do
    ready_status="$(kubectl get clustersecretstore "${name}" -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' 2>/dev/null || true)"
    phase_status="$(kubectl get clustersecretstore "${name}" -o jsonpath='{.status.conditions[?(@.type=="Ready")].message}' 2>/dev/null || true)"

    if [[ "${ready_status}" == "True" ]]; then
      echo "==> ClusterSecretStore/${name} is Ready"
      return 0
    fi

    if [[ "${elapsed}" -ge "${timeout_seconds}" ]]; then
      echo "ERROR: timeout waiting for ClusterSecretStore/${name}"
      echo "ready=${ready_status:-<none>}"
      echo "message=${phase_status:-<none>}"
      kubectl get clustersecretstore "${name}" -o yaml || true
      return 1
    fi

    sleep "${interval_seconds}"
    elapsed=$((elapsed + interval_seconds))
  done
}

echo "==> Phase 3: ESO via ArgoCD"
kubectl apply -f src/argo-apps/infra/eso.yaml
wait_for_app_synced_healthy external-secrets 900

echo "==> Waiting for ESO CRDs"
wait_for_crd_established externalsecrets.external-secrets.io 300
wait_for_crd_established secretstores.external-secrets.io 300
wait_for_crd_established clustersecretstores.external-secrets.io 300

echo "==> Waiting for ESO deployments"
wait_for_deployment_available "${ESO_NS}" external-secrets 300
wait_for_deployment_available "${ESO_NS}" external-secrets-webhook 300
wait_for_deployment_available "${ESO_NS}" external-secrets-cert-controller 300

echo "==> Verifying ESO API registration"
kubectl api-resources | grep external-secrets.io >/dev/null

echo "==> Phase 4: ClusterSecretStore via ArgoCD"
kubectl apply -f src/argo-apps/infra/cluster-secret-store.yaml

# Line in Phase 4
echo "==> Waiting for ArgoCD application: external-secrets-stores to exist"
until kubectl get application external-secrets-stores -n "${ARGO_NS}" >/dev/null 2>&1; do
  sleep 2
done

echo "==> Waiting for ClusterSecretStore resource to appear"
until kubectl get clustersecretstore aws-ssm-store >/dev/null 2>&1; do
  sleep 2
done

echo "==> Applying Reloader via ArgoCD"
kubectl apply -f src/argo-apps/infra/reloader.yaml
wait_for_app_synced_healthy reloader 600

echo "==> Waiting for Reloader deployment"
wait_for_deployment_available "${RELOADER_NS}" reloader-reloader 300

echo
echo "==> Final status"
echo
kubectl get applications -n "${ARGO_NS}"
echo
kubectl get crd | grep external-secrets.io || true
echo
kubectl get clustersecretstore || true
echo
kubectl get pods -n "${ESO_NS}"
echo
kubectl get pods -n "${RELOADER_NS}"