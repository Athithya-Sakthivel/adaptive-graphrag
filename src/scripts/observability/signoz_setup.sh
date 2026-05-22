# src/manifests/signoz/argo-app.yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: signoz
  namespace: argocd
  annotations:
    argocd.argoproj.io/sync-wave: "5"
  finalizers:
    - resources-finalizer.argocd.argoproj.io
spec:
  project: default
  source:
    repoURL: https://github.com/your-org/Autonomous-Incident-Responder.git
    targetRevision: main
    helm:
      values: |
        global:
          storageClass: "default-storage-class"
          clusterDomain: "cluster.local"
          clusterName: "prod-eks"
          cloud: "aws"
        
        clusterName: "prod-eks"
        
        clickhouse:
          enabled: true
          cluster: "cluster"
          database: "signoz_metrics"
          traceDatabase: "signoz_traces"
          logDatabase: "signoz_logs"
          meterDatabase: "signoz_meter"
          user: "admin"
          password: "placeholder"
          layout:
            shardsCount: 1
            replicasCount: 1
          zookeeper:
            enabled: true
            replicaCount: 1
            resources:
              requests:
                cpu: "250m"
                memory: "500Mi"
              limits:
                cpu: "1"
                memory: "1Gi"
          resources:
            requests:
              cpu: "1"
              memory: "1Gi"
            limits:
              cpu: "2"
              memory: "2Gi"
          securityContext:
            enabled: true
            runAsUser: 101
            runAsGroup: 101
            fsGroup: 101
            fsGroupChangePolicy: "OnRootMismatch"
          persistence:
            enabled: true
            existingClaim: ""
            storageClass: "default-storage-class"
            accessModes:
              - ReadWriteOnce
            size: "20Gi"
          initContainers:
            enabled: true
            udf:
              enabled: true
        
        signoz:
          name: "signoz"
          replicaCount: 2
          env:
            signoz_telemetrystore_provider: "clickhouse"
            signoz_include_only_log_namespaces: "inference"
          podSecurityContext:
            fsGroup: 1000
          securityContext:
            allowPrivilegeEscalation: false
            capabilities:
              drop: ["ALL"]
            readOnlyRootFilesystem: true
            runAsNonRoot: true
            runAsUser: 1000
          resources:
            requests:
              cpu: "1"
              memory: "500Mi"
            limits:
              cpu: "2"
              memory: "1Gi"
          persistence:
            enabled: true
            existingClaim: ""
            storageClass: "default-storage-class"
            accessModes:
              - ReadWriteOnce
            size: "10Gi"
          service:
            type: "ClusterIP"
          ingress:
            enabled: false
  destination:
    server: https://kubernetes.default.svc
    namespace: signoz
  syncPolicy:
    automated:
      prune: false
      selfHeal: true
    syncOptions:
      - CreateNamespace=true
      - PrunePropagationPolicy=foreground
      - RespectIgnoreDifferences=true
    retry:
      limit: 5
      backoff:
        duration: 10s
        factor: 2
        maxDuration: 3m
  ignoreDifferences:
    - group: apps
      kind: Deployment
      jsonPointers:
        - /spec/replicas
    - group: apps
      kind: StatefulSet
      jsonPointers:
        - /spec/replicas
    - group: ""
      kind: Secret
      name: signoz-clickhouse-credentials
      jsonPointers:
        - /data/admin-password
  revisionHistoryLimit: 3