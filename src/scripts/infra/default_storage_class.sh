#!/bin/bash
# Usage: 
# bash src/infra/core/default_storage_class.sh k3s 
# bash src/infra/core/default_storage_class.sh eks

TYPE=${1:-k3s}
kubectl delete storageclass --all --ignore-not-found

if [ "$TYPE" = "k3s" ]; then
cat <<EOF | kubectl apply -f -
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: default-storage-class
  annotations:
    storageclass.kubernetes.io/is-default-class: "true"
provisioner: rancher.io/local-path
reclaimPolicy: Delete
volumeBindingMode: WaitForFirstConsumer
EOF
else
cat <<EOF | kubectl apply -f -
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: default-storage-class
  annotations:
    storageclass.kubernetes.io/is-default-class: "true"
provisioner: ebs.csi.aws.com
parameters:
  type: gp3
  encrypted: "true"
reclaimPolicy: Delete
volumeBindingMode: WaitForFirstConsumer
EOF
fi

kubectl get sc