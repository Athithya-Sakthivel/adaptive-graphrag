#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# ESO + Reloader Integration Test
# Uses the REAL SSM parameter: /autonomous-incident-responder/tool-server/api-key
# ============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

PASS=0
FAIL=0

TEST_NAMESPACE="eso-reloader-test"
SSM_PATH="/autonomous-incident-responder/tool-server/api-key"
AWS_REGION="${AWS_REGION:-ap-south-1}"

# ─── Helper functions ────────────────────────────────────────────────────────
cleanup() {
    echo -e "\n${YELLOW}=== Cleanup ===${NC}"
    echo "  Keeping namespace for inspection. Delete manually with:"
    echo "    kubectl delete namespace $TEST_NAMESPACE"
}

pass() { echo -e "${GREEN}✓ $1${NC}"; PASS=$((PASS + 1)); }
fail() { echo -e "${RED}✗ $1${NC}"; FAIL=$((FAIL + 1)); }
log()  { echo -e "  $1"; }

echo -e "${YELLOW}============================================${NC}"
echo -e "${YELLOW}  ESO + Reloader Integration Test${NC}"
echo -e "${YELLOW}  SSM Path: $SSM_PATH${NC}"
echo -e "${YELLOW}============================================${NC}"

# ─── PREREQUISITES ───────────────────────────────────────────────────────────
echo -e "\n${YELLOW}--- Prerequisites ---${NC}"

log "Checking ExternalSecrets CRD..."
kubectl get crd externalsecrets.external-secrets.io >/dev/null 2>&1 \
    && pass "ExternalSecrets CRD exists" \
    || { fail "ExternalSecrets CRD not found"; exit 1; }

log "Checking Reloader deployment in namespace 'reloader'..."
kubectl get deployment -n reloader reloader-reloader >/dev/null 2>&1 \
    && pass "Reloader deployment exists" \
    || { fail "Reloader deployment not found"; exit 1; }

log "Checking ClusterSecretStore 'cluster-secret-store'..."
CSS_STATUS=$(kubectl get clustersecretstore cluster-secret-store \
    -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' 2>/dev/null || echo "False")
log "  ClusterSecretStore Ready: $CSS_STATUS"
[ "$CSS_STATUS" = "True" ] \
    && pass "ClusterSecretStore is Ready" \
    || fail "ClusterSecretStore is NOT Ready"

log "Checking SSM parameter: $SSM_PATH"
SSM_VALUE=$(aws ssm get-parameter --name "$SSM_PATH" --with-decryption \
    --region "$AWS_REGION" --query 'Parameter.Value' --output text 2>/dev/null || echo "")
log "  SSM value length: ${#SSM_VALUE} chars"
[ -n "$SSM_VALUE" ] \
    && pass "SSM parameter exists" \
    || { fail "SSM parameter not found"; exit 1; }

# ─── SETUP ───────────────────────────────────────────────────────────────────
echo -e "\n${YELLOW}--- Setup ---${NC}"

log "Creating test namespace: $TEST_NAMESPACE"
kubectl create namespace "$TEST_NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
pass "Namespace $TEST_NAMESPACE created"

# ─── TEST 1: ExternalSecret Fetches from SSM ─────────────────────────────────
echo -e "\n${YELLOW}--- Test 1: ExternalSecret Fetches from SSM ---${NC}"

cat <<EOF | kubectl apply -f -
apiVersion: external-secrets.io/v1
kind: ExternalSecret
metadata:
  name: test-external-secret
  namespace: $TEST_NAMESPACE
spec:
  refreshInterval: 10s
  secretStoreRef:
    name: cluster-secret-store
    kind: ClusterSecretStore
  target:
    name: test-api-key
    creationPolicy: Owner
  data:
  - secretKey: api-key
    remoteRef:
      key: $SSM_PATH
EOF
pass "ExternalSecret manifest applied"

log "Waiting for ESO to create Kubernetes Secret..."
for i in $(seq 1 20); do
    if kubectl get secret test-api-key -n "$TEST_NAMESPACE" >/dev/null 2>&1; then
        log "Secret 'test-api-key' appeared after $((i * 3))s"
        break
    fi
    sleep 3
done

kubectl get secret test-api-key -n "$TEST_NAMESPACE" >/dev/null 2>&1 \
    && pass "Secret 'test-api-key' created by ESO" \
    || { fail "Secret not created within 60s"; exit 1; }

SECRET_VALUE=$(kubectl get secret test-api-key -n "$TEST_NAMESPACE" \
    -o jsonpath='{.data.api-key}' | base64 -d)
log "  SSM   : ${SSM_VALUE:0:12}..."
log "  Secret: ${SECRET_VALUE:0:12}..."
[ "$SECRET_VALUE" = "$SSM_VALUE" ] \
    && pass "Secret value matches SSM parameter" \
    || fail "Secret value mismatch"

# ─── TEST 2: Deploy Test App with Reloader Annotation ────────────────────────
echo -e "\n${YELLOW}--- Test 2: Deploy Test App with Reloader Annotation ---${NC}"

cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: test-app
  namespace: $TEST_NAMESPACE
  annotations:
    secret.reloader.stakater.com/reload: "test-api-key"
  labels:
    app: test-app
spec:
  replicas: 2
  selector:
    matchLabels:
      app: test-app
  template:
    metadata:
      labels:
        app: test-app
    spec:
      containers:
      - name: app
        image: nginx:alpine
        env:
        - name: API_KEY
          valueFrom:
            secretKeyRef:
              name: test-api-key
              key: api-key
        readinessProbe:
          httpGet:
            path: /
            port: 80
          initialDelaySeconds: 2
          periodSeconds: 2
EOF
pass "Test Deployment manifest applied"

log "Waiting for rollout..."
kubectl rollout status deployment/test-app -n "$TEST_NAMESPACE" --timeout=60s >/dev/null 2>&1 \
    && pass "Rollout complete" \
    || { fail "Rollout failed"; exit 1; }

echo -e "\n${YELLOW}  Current state in $TEST_NAMESPACE:${NC}"
kubectl get all -n "$TEST_NAMESPACE"
kubectl get externalsecret -n "$TEST_NAMESPACE"

ORIGINAL_PODS=$(kubectl get pods -n "$TEST_NAMESPACE" -l app=test-app -o jsonpath='{.items[*].metadata.name}')
log "Pods: $ORIGINAL_PODS"

SAMPLE_POD=$(echo "$ORIGINAL_PODS" | awk '{print $1}')
POD_VALUE=$(kubectl exec -n "$TEST_NAMESPACE" "$SAMPLE_POD" -- printenv API_KEY 2>/dev/null || echo "EXEC_FAILED")
log "  Pod $SAMPLE_POD API_KEY: ${POD_VALUE:0:12}..."
[ "$POD_VALUE" = "$SSM_VALUE" ] \
    && pass "Pod env matches SSM value" \
    || fail "Pod env mismatch"

# ─── TEST 3: Update SSM → ESO Syncs → Reloader Restarts ─────────────────────
echo -e "\n${YELLOW}--- Test 3: End-to-End Secret Rotation ---${NC}"

NEW_VALUE="rotated-$(date +%s)-$(openssl rand -hex 4)"
log "Updating SSM to: $NEW_VALUE"
aws ssm put-parameter --name "$SSM_PATH" --value "$NEW_VALUE" \
    --type "SecureString" --region "$AWS_REGION" --overwrite >/dev/null 2>&1
pass "SSM parameter updated"

log "Waiting 30s for ESO sync + Reloader detection..."
sleep 30

echo -e "\n${YELLOW}  Post-update state in $TEST_NAMESPACE:${NC}"
kubectl get pods -n "$TEST_NAMESPACE" -l app=test-app -o wide
kubectl get secret test-api-key -n "$TEST_NAMESPACE" -o jsonpath='{.data.api-key}' | base64 -d
echo ""

UPDATED_SECRET=$(kubectl get secret test-api-key -n "$TEST_NAMESPACE" \
    -o jsonpath='{.data.api-key}' | base64 -d)
log "  Secret value: ${UPDATED_SECRET:0:12}..."
[ "$UPDATED_SECRET" = "$NEW_VALUE" ] \
    && pass "ESO synced new value to Secret" \
    || fail "ESO sync failed. Expected: $NEW_VALUE, Got: $UPDATED_SECRET"

NEW_PODS=$(kubectl get pods -n "$TEST_NAMESPACE" -l app=test-app -o jsonpath='{.items[*].metadata.name}')
log "  Original pods: $ORIGINAL_PODS"
log "  Current pods:  $NEW_PODS"

RESTART_COUNT=$(kubectl get pods -n "$TEST_NAMESPACE" -l app=test-app \
    -o jsonpath='{.items[*].status.containerStatuses[0].restartCount}')
log "  Restart counts: $RESTART_COUNT"

# Check if Reloader did its job
RELOADER_WORKED=false
if [ "$ORIGINAL_PODS" != "$NEW_PODS" ]; then
    RELOADER_WORKED=true
    log "  Pod names changed → Reloader created new ReplicaSet"
fi
for count in $RESTART_COUNT; do
    if [ "$count" -gt 0 ] 2>/dev/null; then
        RELOADER_WORKED=true
        log "  Container restarted $count times → Reloader triggered restart"
    fi
done

[ "$RELOADER_WORKED" = true ] \
    && pass "Reloader triggered rolling restart" \
    || fail "Reloader did NOT trigger restart"

log "Verifying new pods have rotated secret..."
kubectl rollout status deployment/test-app -n "$TEST_NAMESPACE" --timeout=60s >/dev/null 2>&1
FRESH_POD=$(kubectl get pods -n "$TEST_NAMESPACE" -l app=test-app -o jsonpath='{.items[0].metadata.name}')
FRESH_VALUE=$(kubectl exec -n "$TEST_NAMESPACE" "$FRESH_POD" -- printenv API_KEY 2>/dev/null || echo "EXEC_FAILED")
log "  Pod $FRESH_POD API_KEY: ${FRESH_VALUE:0:12}..."
[ "$FRESH_VALUE" = "$NEW_VALUE" ] \
    && pass "New pods have rotated secret" \
    || fail "Pods have wrong secret. Expected: $NEW_VALUE, Got: $FRESH_VALUE"

# ─── RESTORE ─────────────────────────────────────────────────────────────────
echo -e "\n${YELLOW}--- Restore Original SSM Value ---${NC}"
aws ssm put-parameter --name "$SSM_PATH" --value "$SSM_VALUE" \
    --type "SecureString" --region "$AWS_REGION" --overwrite >/dev/null 2>&1
pass "SSM parameter restored"

# ─── SUMMARY ─────────────────────────────────────────────────────────────────
echo -e "\n${YELLOW}============================================${NC}"
echo -e "${YELLOW}  Results: ${GREEN}$PASS passed${NC}, ${RED}$FAIL failed${NC}"
echo -e "${YELLOW}============================================${NC}"

if [ "$FAIL" -gt 0 ]; then
    echo -e "\n${RED}FAILED. Debug commands:${NC}"
    echo "  kubectl -n $TEST_NAMESPACE describe externalsecret test-external-secret"
    echo "  kubectl -n external-secrets logs deployment/external-secrets --tail=50"
    echo "  kubectl -n reloader logs deployment/reloader-reloader --tail=50"
    echo "  kubectl -n $TEST_NAMESPACE get all"
    exit 1
else
    echo -e "\n${GREEN}All $PASS tests passed.${NC}"
    kubectl delete namespace eso-reloader-test
fi