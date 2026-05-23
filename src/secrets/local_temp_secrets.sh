echo "==> Creating aws-creds secret required by ESO controller if using k3s"
kubectl -n external-secrets create secret generic aws-creds \
  --from-literal=access-key-id="${AWS_ACCESS_KEY_ID}" \
  --from-literal=secret-access-key="${AWS_SECRET_ACCESS_KEY}" \
  --from-literal=session-token="${AWS_SESSION_TOKEN:-}" \
  --from-literal=region="${AWS_REGION}" \
  --dry-run=client -o yaml | kubectl apply -f -
  


kubectl create secret generic clickhouse-credentials -n logging \
    --from-literal=username="${CLICKHOUSE_USER:-vector}" \
    --from-literal=password="${CLICKHOUSE_PASSWORD:-vectorpass}" \
    --dry-run=client -o yaml | kubectl apply -f -