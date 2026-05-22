# Secrets Management (ESO + Reloader)

## Goal

Provide Kubernetes workloads with centrally managed secrets that support:

* automatic rotation
* GitOps deployment
* low application coupling
* controlled AWS API usage
* zero manual pod restarts

---

# Why This Architecture

Direct application reads from AWS SSM are intentionally avoided.

SSM Parameter Store has request rate limits and API latency. If every pod reads secrets directly from AWS:

* startup becomes dependent on AWS availability
* scaling events amplify API calls
* secret refreshes increase AWS traffic
* outages propagate into workloads
* retry storms become possible during incidents

Instead, the cluster uses Kubernetes Secrets as a local cache layer.

Mental model:

```text
AWS SSM
  = source of truth

ESO
  = synchronization controller

Kubernetes Secret
  = local cached runtime copy

Reloader
  = rollout trigger
```

Applications only read Kubernetes Secrets.

Applications never communicate with AWS SSM directly.

---

# Architecture

```text
AWS SSM Parameter Store
        ↓
External Secrets Operator (ESO)
        ↓
Kubernetes Secret
        ↓
Deployment / Pod
        ↓
Reloader watches Secret changes
        ↓
Rolling restart triggered
```

---

# Components

## External Secrets Operator (ESO)

Purpose:

* reads secrets from AWS SSM
* synchronizes them into Kubernetes Secrets
* periodically refreshes values

Namespace:

```text
external-secrets
```

API version used:

```text
external-secrets.io/v1
```

The cluster does not use deprecated `v1beta1` resources.

---

## ClusterSecretStore

Purpose:

* shared AWS backend configuration
* centralized authentication configuration for ESO

Resource:

```yaml
apiVersion: external-secrets.io/v1
kind: ClusterSecretStore
```

Backend:

```text
AWS SSM Parameter Store
Region: ap-south-1
```

Store name:

```text
cluster-secret-store
```

---

## Reloader

Purpose:

Kubernetes does not automatically restart pods when Secrets change.

Reloader watches Secrets and ConfigMaps and triggers rolling restarts for annotated workloads.

Without Reloader:

```text
Secret changes
    ↓
running pods keep stale values
```

With Reloader:

```text
Secret changes
    ↓
new ReplicaSet created
    ↓
pods restarted automatically
```

Namespace:

```text
reloader
```

Deployment:

```text
reloader-reloader
```

---

# Secret Lifecycle

## 1. Secret Stored in AWS SSM

Example:

```text
/autonomous-incident-responder/tool-server/api-key
```

AWS SSM remains the source of truth.

---

## 2. ESO Synchronizes Secret

Example `ExternalSecret`:

```yaml
apiVersion: external-secrets.io/v1
kind: ExternalSecret
metadata:
  name: tool-server-secret

spec:
  refreshInterval: 10s

  secretStoreRef:
    name: cluster-secret-store
    kind: ClusterSecretStore

  target:
    name: tool-server-secret

  data:
    - secretKey: API_KEY
      remoteRef:
        key: /autonomous-incident-responder/tool-server/api-key
```

ESO periodically fetches the value from SSM and writes it into a Kubernetes Secret.

---

## 3. Application Consumes Kubernetes Secret

Example:

```yaml
env:
  - name: API_KEY
    valueFrom:
      secretKeyRef:
        name: tool-server-secret
        key: API_KEY
```

Applications only read Kubernetes-native Secrets.

No AWS SDK calls are required inside workloads.

---

## 4. Secret Rotation

When the SSM parameter changes:

```text
SSM updated
    ↓
ESO refresh detects change
    ↓
Kubernetes Secret updated
    ↓
Reloader detects Secret modification
    ↓
Deployment restarted
    ↓
New pods receive new value
```

Rotation becomes automatic and operationally predictable.

---

# Why Kubernetes Secrets Are Used As Cache

This design intentionally trades eventual consistency for stability.

Benefits:

| Benefit                       | Reason                                               |
| ----------------------------- | ---------------------------------------------------- |
| Lower SSM API usage           | workloads do not repeatedly call AWS                 |
| Faster pod startup            | secrets are local to cluster                         |
| Better resilience             | pods do not depend on live AWS calls                 |
| Reduced blast radius          | AWS throttling does not directly impact applications |
| Native Kubernetes integration | standard Secret refs and env vars                    |
| Predictable rotation          | ESO + Reloader handle lifecycle                      |

---

# Operational Rules

## Applications Must Never Read SSM Directly

Applications consume only:

* Kubernetes Secrets
* mounted Secret volumes
* environment variables

This prevents:

* credential sprawl
* duplicated AWS logic
* inconsistent refresh behavior
* uncontrolled AWS API usage

---

## All Rotatable Workloads Must Use Reloader

Required annotation:

```yaml
metadata:
  annotations:
    reloader.stakater.com/auto: "true"
```

Without this annotation, pods will not restart after secret updates.

---

## Only ESO Communicates With AWS

AWS access is centralized inside ESO.

This simplifies:

* IAM management
* auditability
* rate limiting
* operational debugging

---

# Validation

Integration test:

```text
src/tests/infra_tests/eso_reloader_tests.sh
```

Validated behaviors:

* ESO can read SSM
* Kubernetes Secret is created
* Secret matches SSM value
* Workload receives secret
* Secret rotation propagates correctly
* Reloader triggers rolling restart
* New pods receive rotated value

Expected successful result:

```text
16 passed, 0 failed
```

---

# Common Failure Modes

## `external-secrets.io/v1beta1` not found

Cause:

Old API version.

Fix:

```yaml
apiVersion: external-secrets.io/v1
```

---

## Pods do not restart after rotation

Cause:

Missing Reloader annotation.

Required:

```yaml
reloader.stakater.com/auto: "true"
```

---

## Secret not syncing

Check:

* ESO pods
* ClusterSecretStore Ready status
* AWS credentials
* SSM parameter path
* refresh interval

---

# Verification Commands

## Check ESO

```bash
kubectl get pods -n external-secrets
kubectl get crd | grep external-secrets.io
```

## Check Reloader

```bash
kubectl get pods -n reloader
kubectl get deployment -n reloader
```

## Check Secret Store

```bash
kubectl get clustersecretstore
kubectl describe clustersecretstore cluster-secret-store
```

## Check Secret

```bash
kubectl get secrets -A
```

## Check Pod Environment

```bash
kubectl exec deploy/<deployment> -- printenv | grep API_KEY
```
