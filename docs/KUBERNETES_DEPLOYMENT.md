# Kubernetes Deployment Guide

This guide provides comprehensive instructions for deploying Workspace Qdrant MCP on Kubernetes with enterprise-grade security, scaling, and monitoring.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Prerequisites](#prerequisites)
3. [Cluster Preparation](#cluster-preparation)
4. [Installation](#installation)
5. [Configuration](#configuration)
6. [Scaling and High Availability](#scaling-and-high-availability)
7. [Security](#security)
8. [Monitoring and Observability](#monitoring-and-observability)
9. [Maintenance](#maintenance)
10. [Troubleshooting](#troubleshooting)
11. [Advanced Configurations](#advanced-configurations)

## Quick Start

### Single Command Deployment

```bash
# Deploy with default configuration
kubectl apply -k docker/k8s/

# Verify deployment
kubectl get pods -n workspace-qdrant-mcp
kubectl get services -n workspace-qdrant-mcp
```

### Access Application

```bash
# Port forward for testing
kubectl port-forward -n workspace-qdrant-mcp service/workspace-qdrant-mcp-service 8000:8000

# Test connection
curl http://localhost:8000/health
```

## Prerequisites

### Kubernetes Cluster Requirements

- **Kubernetes Version**: 1.24+
- **Node Requirements**: 
  - Minimum 3 nodes (for HA)
  - 4+ CPU cores per node
  - 8GB+ RAM per node
  - 100GB+ storage per node
- **Storage**: CSI-compatible storage class
- **Network**: CNI plugin with NetworkPolicy support

### Required Kubernetes Components

```bash
# Verify cluster readiness
kubectl cluster-info
kubectl get nodes
kubectl get storageclass

# Required components
kubectl get deployment -n kube-system metrics-server
kubectl get deployment -n ingress-nginx ingress-nginx-controller
```

### Tools Installation

```bash
# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# Install kustomize
curl -s "https://raw.githubusercontent.com/kubernetes-sigs/kustomize/master/hack/install_kustomize.sh" | bash
sudo mv kustomize /usr/local/bin/

# Install Helm (optional)
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
```

## Cluster Preparation

### Namespace Setup

```bash
# Create namespace
kubectl apply -f docker/k8s/namespace.yaml

# Verify namespace
kubectl get namespace workspace-qdrant-mcp
kubectl describe namespace workspace-qdrant-mcp
```

### Storage Configuration

```yaml
# storage-config.yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: fast-ssd
  annotations:
    storageclass.kubernetes.io/is-default-class: "false"
provisioner: kubernetes.io/aws-ebs  # Adjust for your cloud provider
parameters:
  type: gp3
  iops: "3000"
  throughput: "125"
  fsType: ext4
  encrypted: "true"
allowVolumeExpansion: true
volumeBindingMode: WaitForFirstConsumer
reclaimPolicy: Retain
```

```bash
# Apply storage configuration
kubectl apply -f storage-config.yaml
kubectl get storageclass
```

### Network Policies

```bash
# Apply network policies
kubectl apply -f docker/k8s/namespace.yaml

# Verify network policies
kubectl get networkpolicy -n workspace-qdrant-mcp
```

## Installation

### Method 1: Direct YAML Application

```bash
# Apply all manifests
kubectl apply -f docker/k8s/namespace.yaml
kubectl apply -f docker/k8s/configmap.yaml
kubectl apply -f docker/k8s/secrets.yaml
kubectl apply -f docker/k8s/persistentvolume.yaml
kubectl apply -f docker/k8s/rbac.yaml
kubectl apply -f docker/k8s/deployment.yaml
kubectl apply -f docker/k8s/service.yaml
kubectl apply -f docker/k8s/ingress.yaml
kubectl apply -f docker/k8s/hpa.yaml
```

### Method 2: Kustomization

```yaml
# kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

resources:
- docker/k8s/namespace.yaml
- docker/k8s/configmap.yaml
- docker/k8s/secrets.yaml
- docker/k8s/persistentvolume.yaml
- docker/k8s/rbac.yaml
- docker/k8s/deployment.yaml
- docker/k8s/service.yaml
- docker/k8s/ingress.yaml
- docker/k8s/hpa.yaml

images:
- name: workspace-qdrant-mcp
  newTag: latest

patchesStrategicMerge:
- production-patches.yaml
```

```bash
# Deploy with kustomize
kubectl apply -k .
```

### Method 3: Helm Chart (Optional)

```bash
# Create Helm chart structure
mkdir -p helm/workspace-qdrant-mcp/{templates,charts}
cp docker/k8s/*.yaml helm/workspace-qdrant-mcp/templates/

# Install with Helm
helm install workspace-qdrant-mcp ./helm/workspace-qdrant-mcp \
  --namespace workspace-qdrant-mcp \
  --create-namespace
```

## Configuration

### Secrets Management

```bash
# Create secrets from files
kubectl create secret generic workspace-qdrant-mcp-secrets \
  --from-literal=QDRANT_API_KEY="your-secure-api-key" \
  --from-literal=REDIS_PASSWORD="your-secure-password" \
  --from-literal=APP_SECRET_KEY="your-app-secret" \
  --from-literal=JWT_SECRET="your-jwt-secret" \
  -n workspace-qdrant-mcp

# Create TLS secret
kubectl create secret tls workspace-qdrant-mcp-tls \
  --cert=path/to/tls.crt \
  --key=path/to/tls.key \
  -n workspace-qdrant-mcp

# Create registry secret (if using private registry)
kubectl create secret docker-registry registry-secret \
  --docker-server=your-registry.com \
  --docker-username=your-username \
  --docker-password=your-password \
  --docker-email=your-email \
  -n workspace-qdrant-mcp
```

### ConfigMap Updates

```bash
# Update application configuration
kubectl create configmap workspace-qdrant-mcp-config \
  --from-literal=WORKSPACE_QDRANT_LOG_LEVEL="INFO" \
  --from-literal=QDRANT_HOST="qdrant-service" \
  --from-literal=REDIS_HOST="redis-service" \
  -n workspace-qdrant-mcp \
  --dry-run=client -o yaml | kubectl apply -f -

# Restart deployments to pick up changes
kubectl rollout restart deployment/workspace-qdrant-mcp -n workspace-qdrant-mcp
```

### Environment-Specific Configuration

```yaml
# production-overlay/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

resources:
- ../base

patchesStrategicMerge:
- replica-count.yaml
- resource-limits.yaml

configMapGenerator:
- name: workspace-qdrant-mcp-config
  behavior: merge
  literals:
  - WORKSPACE_QDRANT_LOG_LEVEL=INFO
  - WORKSPACE_QDRANT_ENV=production
```

## Scaling and High Availability

### Horizontal Pod Autoscaler

```bash
# Check HPA status
kubectl get hpa -n workspace-qdrant-mcp

# Describe HPA for details
kubectl describe hpa workspace-qdrant-mcp-hpa -n workspace-qdrant-mcp

# Manual scaling
kubectl scale deployment workspace-qdrant-mcp --replicas=5 -n workspace-qdrant-mcp
```

### Vertical Pod Autoscaler

```yaml
# Enable VPA (if available)
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: workspace-qdrant-mcp-vpa
  namespace: workspace-qdrant-mcp
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: workspace-qdrant-mcp
  updatePolicy:
    updateMode: "Auto"
```

### Pod Disruption Budget

```bash
# Check PDB status
kubectl get pdb -n workspace-qdrant-mcp
kubectl describe pdb workspace-qdrant-mcp-pdb -n workspace-qdrant-mcp

# Test disruption
kubectl drain <node-name> --ignore-daemonsets --delete-emptydir-data
```

### Multi-Zone Deployment

```yaml
# Add node affinity for multi-zone deployment
spec:
  affinity:
    podAntiAffinity:
      preferredDuringSchedulingIgnoredDuringExecution:
      - weight: 100
        podAffinityTerm:
          labelSelector:
            matchExpressions:
            - key: app
              operator: In
              values:
              - workspace-qdrant-mcp
          topologyKey: topology.kubernetes.io/zone
```

## Security

### RBAC Configuration

```bash
# Check service accounts
kubectl get serviceaccount -n workspace-qdrant-mcp

# Check roles and bindings
kubectl get role,rolebinding -n workspace-qdrant-mcp

# Test permissions
kubectl auth can-i get pods --as=system:serviceaccount:workspace-qdrant-mcp:workspace-qdrant-mcp -n workspace-qdrant-mcp
```

### Network Security

```bash
# Check network policies
kubectl get networkpolicy -n workspace-qdrant-mcp

# Test network connectivity
kubectl run test-pod --image=busybox -n workspace-qdrant-mcp --rm -it -- /bin/sh
# Inside the pod:
# wget -qO- http://workspace-qdrant-mcp-service:8000/health
```

### Pod Security Standards

```yaml
# Add pod security context
spec:
  securityContext:
    runAsNonRoot: true
    runAsUser: 65534
    runAsGroup: 65534
    fsGroup: 65534
    seccompProfile:
      type: RuntimeDefault
  containers:
  - name: workspace-qdrant-mcp
    securityContext:
      allowPrivilegeEscalation: false
      readOnlyRootFilesystem: true
      capabilities:
        drop:
        - ALL
        add:
        - NET_BIND_SERVICE
```

### Secrets Scanning

```bash
# Scan for secrets in config
kubectl get secrets -n workspace-qdrant-mcp -o yaml | grep -E "(password|key|token)" || echo "No plain-text secrets found"

# Use external secret management (example with External Secrets Operator)
kubectl apply -f - <<EOF
apiVersion: external-secrets.io/v1beta1
kind: SecretStore
metadata:
  name: vault-backend
  namespace: workspace-qdrant-mcp
spec:
  provider:
    vault:
      server: "https://vault.example.com"
      path: "secret"
      version: "v2"
EOF
```

## Monitoring and Observability

### Prometheus Integration

```bash
# Check ServiceMonitor
kubectl get servicemonitor -n workspace-qdrant-mcp

# Test metrics endpoint
kubectl port-forward -n workspace-qdrant-mcp service/workspace-qdrant-mcp-service 8000:8000
curl http://localhost:8000/metrics
```

### Grafana Dashboards

```bash
# Access Grafana
kubectl port-forward -n workspace-qdrant-mcp service/grafana-service 3000:3000

# Import dashboards (URLs for import)
# - Kubernetes Cluster Dashboard: 315
# - Pod Overview: 747  
# - Application Dashboard: Custom (create based on metrics)
```

### Logging with Loki

```yaml
# Configure Promtail for Kubernetes
apiVersion: v1
kind: ConfigMap
metadata:
  name: promtail-config
data:
  promtail.yml: |
    server:
      http_listen_port: 3101
    
    clients:
      - url: http://loki:3100/loki/api/v1/push
    
    scrape_configs:
      - job_name: kubernetes-pods
        kubernetes_sd_configs:
          - role: pod
        relabel_configs:
          - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
            action: keep
            regex: true
```

### Distributed Tracing

```bash
# Check Jaeger deployment
kubectl get deployment jaeger -n workspace-qdrant-mcp

# Access Jaeger UI
kubectl port-forward -n workspace-qdrant-mcp service/jaeger-service 16686:16686

# Test trace collection
curl -X POST http://localhost:8000/api/test-trace
```

## Maintenance

### Rolling Updates

```bash
# Update image version
kubectl set image deployment/workspace-qdrant-mcp workspace-qdrant-mcp=workspace-qdrant-mcp:v0.2.1 -n workspace-qdrant-mcp

# Check rollout status
kubectl rollout status deployment/workspace-qdrant-mcp -n workspace-qdrant-mcp

# Rollback if needed
kubectl rollout undo deployment/workspace-qdrant-mcp -n workspace-qdrant-mcp
```

### Backup Operations

```yaml
# Backup CronJob
apiVersion: batch/v1
kind: CronJob
metadata:
  name: backup-job
  namespace: workspace-qdrant-mcp
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: backup
            image: alpine:latest
            command:
            - /bin/sh
            - -c
            - |
              apk add --no-cache curl tar
              kubectl exec deployment/qdrant -- tar czf - /qdrant/storage > /backup/qdrant-$(date +%Y%m%d).tar.gz
              # Upload to S3 or other backup storage
          restartPolicy: OnFailure
```

### Capacity Planning

```bash
# Check resource usage
kubectl top nodes
kubectl top pods -n workspace-qdrant-mcp

# Check storage usage
kubectl get pv
kubectl describe pvc -n workspace-qdrant-mcp

# Resource recommendations
kubectl describe vpa workspace-qdrant-mcp-vpa -n workspace-qdrant-mcp
```

### Certificate Management

```bash
# Check certificate expiration
kubectl get certificate -n workspace-qdrant-mcp
kubectl describe certificate workspace-qdrant-mcp-tls -n workspace-qdrant-mcp

# Renew certificates (cert-manager)
kubectl annotate certificate workspace-qdrant-mcp-tls cert-manager.io/issue-temporary-certificate="true" -n workspace-qdrant-mcp
```

## Troubleshooting

### Common Issues

#### Pods Not Starting

```bash
# Check pod status
kubectl get pods -n workspace-qdrant-mcp
kubectl describe pod <pod-name> -n workspace-qdrant-mcp

# Check logs
kubectl logs <pod-name> -n workspace-qdrant-mcp
kubectl logs <pod-name> -n workspace-qdrant-mcp --previous

# Check events
kubectl get events -n workspace-qdrant-mcp --sort-by='.lastTimestamp'
```

#### Service Discovery Issues

```bash
# Test service resolution
kubectl run debug-pod --image=busybox -n workspace-qdrant-mcp --rm -it -- /bin/sh
# nslookup workspace-qdrant-mcp-service
# nslookup qdrant-service

# Check service endpoints
kubectl get endpoints -n workspace-qdrant-mcp
kubectl describe service workspace-qdrant-mcp-service -n workspace-qdrant-mcp
```

#### Persistent Volume Issues

```bash
# Check PV/PVC status
kubectl get pv
kubectl get pvc -n workspace-qdrant-mcp

# Check storage class
kubectl get storageclass
kubectl describe storageclass fast-ssd

# Force PVC deletion (if stuck)
kubectl patch pvc <pvc-name> -n workspace-qdrant-mcp -p '{"metadata":{"finalizers":null}}'
```

#### Ingress Issues

```bash
# Check ingress status
kubectl get ingress -n workspace-qdrant-mcp
kubectl describe ingress workspace-qdrant-mcp-ingress -n workspace-qdrant-mcp

# Check ingress controller logs
kubectl logs -n ingress-nginx deployment/ingress-nginx-controller

# Test ingress backend
kubectl port-forward -n workspace-qdrant-mcp service/workspace-qdrant-mcp-service 8080:8000
curl http://localhost:8080/health
```

### Debugging Tools

```bash
# Debug container
kubectl run debug-pod --image=nicolaka/netshoot -n workspace-qdrant-mcp --rm -it -- /bin/bash

# Exec into running pod
kubectl exec -it deployment/workspace-qdrant-mcp -n workspace-qdrant-mcp -- /bin/bash

# Port forwarding for debugging
kubectl port-forward -n workspace-qdrant-mcp pod/<pod-name> 5678:5678  # Debug port

# Resource monitoring
watch kubectl top pods -n workspace-qdrant-mcp
```

### Performance Troubleshooting

```bash
# Check HPA metrics
kubectl get hpa workspace-qdrant-mcp-hpa -n workspace-qdrant-mcp -o yaml

# Check custom metrics
kubectl get --raw "/apis/custom.metrics.k8s.io/v1beta1"

# Cluster autoscaler logs
kubectl logs -n kube-system deployment/cluster-autoscaler
```

## Advanced Configurations

### Service Mesh Integration (Istio)

```yaml
# Enable sidecar injection
apiVersion: v1
kind: Namespace
metadata:
  name: workspace-qdrant-mcp
  labels:
    istio-injection: enabled
```

```yaml
# Virtual Service
apiVersion: networking.istio.io/v1alpha3
kind: VirtualService
metadata:
  name: workspace-qdrant-mcp-vs
  namespace: workspace-qdrant-mcp
spec:
  hosts:
  - workspace-qdrant-mcp.example.com
  gateways:
  - workspace-qdrant-mcp-gateway
  http:
  - route:
    - destination:
        host: workspace-qdrant-mcp-service
        port:
          number: 8000
```

### GitOps with ArgoCD

```yaml
# Application manifest
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: workspace-qdrant-mcp
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://github.com/ChrisGVE/workspace-qdrant-mcp
    targetRevision: HEAD
    path: docker/k8s
  destination:
    server: https://kubernetes.default.svc
    namespace: workspace-qdrant-mcp
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
```

### Chaos Engineering

```yaml
# Chaos Mesh experiment
apiVersion: chaos-mesh.org/v1alpha1
kind: PodChaos
metadata:
  name: workspace-qdrant-mcp-kill
  namespace: workspace-qdrant-mcp
spec:
  selector:
    labelSelectors:
      app: workspace-qdrant-mcp
  mode: one
  action: pod-kill
  duration: "30s"
```

### Multi-Cluster Deployment

```bash
# Install Submariner for multi-cluster networking
curl https://get.submariner.io | bash
subctl deploy-broker --kubeconfig cluster1-config.yaml
subctl join --kubeconfig cluster2-config.yaml broker-info.subm
```

---

For additional troubleshooting and advanced scenarios, refer to the [Container Troubleshooting Guide](CONTAINER_TROUBLESHOOTING.md) and the official Kubernetes documentation.