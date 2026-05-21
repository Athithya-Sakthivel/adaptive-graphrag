echo "==> Creating aws-creds secret required by ESO controller"
kubectl -n external-secrets create secret generic aws-creds \
  --from-literal=access-key-id="${AWS_ACCESS_KEY_ID}" \
  --from-literal=secret-access-key="${AWS_SECRET_ACCESS_KEY}" \
  --from-literal=session-token="${AWS_SESSION_TOKEN:-}" \
  --from-literal=region="${AWS_REGION}" \
  --dry-run=client -o yaml | kubectl apply -f -
  