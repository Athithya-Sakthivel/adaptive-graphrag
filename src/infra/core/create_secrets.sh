
aws ssm put-parameter \
    --name "/autonomous-incident-responder/inference/postgres/connection-string-pooler" \
    --value "postgresql://$(kubectl get secret postgres-cluster-app -o jsonpath='{.data.username}' | base64 -d):$(kubectl get secret postgres-cluster-app -o jsonpath='{.data.password}' | base64 -d)@postgres-pooler.default.svc.cluster.local:5432/$(kubectl get secret postgres-cluster-app -o jsonpath='{.data.dbname}' | base64 -d)" \
    --type "SecureString" \
    --region $AWS_REGION \
    --overwrite
