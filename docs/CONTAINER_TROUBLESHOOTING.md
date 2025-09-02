# Container Troubleshooting Guide

This comprehensive troubleshooting guide covers common issues, diagnostic procedures, and solutions for Workspace Qdrant MCP container deployments.

## Table of Contents

1. [Quick Diagnostics](#quick-diagnostics)
2. [Application Issues](#application-issues)
3. [Database Issues](#database-issues)
4. [Network Issues](#network-issues)
5. [Performance Issues](#performance-issues)
6. [Security Issues](#security-issues)
7. [Storage Issues](#storage-issues)
8. [Monitoring Issues](#monitoring-issues)
9. [Kubernetes-Specific Issues](#kubernetes-specific-issues)
10. [Recovery Procedures](#recovery-procedures)

## Quick Diagnostics

### Health Check Commands

```bash
# Docker Compose
docker-compose -f docker/docker-compose.yml ps
docker-compose -f docker/docker-compose.yml logs --tail=50

# Kubernetes
kubectl get pods -n workspace-qdrant-mcp
kubectl describe pod <pod-name> -n workspace-qdrant-mcp

# Application Health
curl http://localhost:8000/health
curl http://localhost:6333/health  # Qdrant
curl http://localhost:6379  # Redis (should return error but connection works)
```

### System Resource Check

```bash
# Docker
docker stats --no-stream
docker system df

# Kubernetes
kubectl top nodes
kubectl top pods -n workspace-qdrant-mcp

# System resources
free -h
df -h
iostat -x 1 5
```

### Log Collection

```bash
# Collect all logs
mkdir -p /tmp/logs/$(date +%Y%m%d-%H%M%S)
cd /tmp/logs/$(date +%Y%m%d-%H%M%S)

# Docker logs
docker-compose -f docker/docker-compose.yml logs --no-color > docker-compose.log
docker logs workspace-qdrant-mcp > app.log 2>&1
docker logs workspace-qdrant-qdrant > qdrant.log 2>&1
docker logs workspace-qdrant-redis > redis.log 2>&1

# Kubernetes logs
kubectl logs -n workspace-qdrant-mcp deployment/workspace-qdrant-mcp > k8s-app.log
kubectl logs -n workspace-qdrant-mcp deployment/qdrant > k8s-qdrant.log
kubectl logs -n workspace-qdrant-mcp deployment/redis > k8s-redis.log

# System logs
journalctl --since "1 hour ago" > system.log
dmesg > dmesg.log
```

## Application Issues

### Application Won't Start

**Symptoms:**
- Container exits immediately
- Health checks fail
- No response on port 8000

**Diagnostic Steps:**

```bash
# Check container status
docker ps -a | grep workspace-qdrant-mcp
kubectl get pods -n workspace-qdrant-mcp -o wide

# Examine logs
docker logs workspace-qdrant-mcp --tail=100
kubectl logs -n workspace-qdrant-mcp deployment/workspace-qdrant-mcp --tail=100

# Check configuration
docker exec workspace-qdrant-mcp env | grep -E "(QDRANT|REDIS|WORKSPACE)"
kubectl exec deployment/workspace-qdrant-mcp -n workspace-qdrant-mcp -- env | grep -E "(QDRANT|REDIS|WORKSPACE)"
```

**Common Solutions:**

```bash
# Fix environment variables
export QDRANT_HOST=qdrant  # Not localhost in container
export REDIS_HOST=redis
export WORKSPACE_QDRANT_LOG_LEVEL=DEBUG

# Check dependencies are running
docker-compose exec qdrant curl http://localhost:6333/health
docker-compose exec redis redis-cli ping

# Verify file permissions
docker exec workspace-qdrant-mcp ls -la /app/data /app/logs
kubectl exec deployment/workspace-qdrant-mcp -n workspace-qdrant-mcp -- ls -la /app/data /app/logs
```

### Import/Module Errors

**Symptoms:**
- ModuleNotFoundError
- ImportError
- Python path issues

**Diagnostic Steps:**

```bash
# Check Python environment
docker exec workspace-qdrant-mcp python -c "import sys; print('\n'.join(sys.path))"
docker exec workspace-qdrant-mcp pip list | grep -E "(qdrant|fastmcp|workspace)"

# Test manual import
docker exec workspace-qdrant-mcp python -c "
import workspace_qdrant_mcp
print(f'Version: {workspace_qdrant_mcp.__version__}')
"
```

**Solutions:**

```bash
# Rebuild with clean environment
docker-compose down
docker-compose build --no-cache workspace-qdrant-mcp
docker-compose up -d

# Fix PYTHONPATH
export PYTHONPATH=/app/src:$PYTHONPATH
```

### Configuration Issues

**Symptoms:**
- Invalid configuration errors
- Feature not working as expected
- Authentication failures

**Diagnostic Steps:**

```bash
# Validate configuration
docker exec workspace-qdrant-mcp python -c "
from workspace_qdrant_mcp.core.config import get_settings
settings = get_settings()
print('Configuration loaded successfully')
print(f'Qdrant Host: {settings.qdrant_host}')
print(f'Redis Host: {settings.redis_host}')
"

# Test configuration validator
docker exec workspace-qdrant-mcp workspace-qdrant-validate --config /app/config
```

**Solutions:**

```yaml
# Fix docker-compose.yml
services:
  workspace-qdrant-mcp:
    environment:
      - QDRANT_HOST=qdrant
      - QDRANT_PORT=6333
      - QDRANT_API_KEY=${QDRANT_API_KEY}
      - REDIS_HOST=redis
      - REDIS_PORT=6379
      - REDIS_PASSWORD=${REDIS_PASSWORD}
```

## Database Issues

### Qdrant Connection Issues

**Symptoms:**
- Connection refused errors
- Timeout errors
- Authentication failures

**Diagnostic Steps:**

```bash
# Check Qdrant status
curl http://localhost:6333/health
curl http://localhost:6333/collections

# Test from application container
docker exec workspace-qdrant-mcp curl http://qdrant:6333/health
kubectl exec deployment/workspace-qdrant-mcp -n workspace-qdrant-mcp -- curl http://qdrant-service:6333/health

# Check Qdrant logs
docker logs workspace-qdrant-qdrant --tail=50
kubectl logs deployment/qdrant -n workspace-qdrant-mcp --tail=50
```

**Solutions:**

```bash
# Restart Qdrant
docker-compose restart qdrant
kubectl rollout restart deployment/qdrant -n workspace-qdrant-mcp

# Check API key
curl -H "api-key: your-api-key" http://localhost:6333/collections

# Reset Qdrant data (WARNING: This deletes all data)
docker-compose down
docker volume rm workspace-qdrant-mcp_qdrant_storage
docker-compose up -d
```

### Redis Connection Issues

**Symptoms:**
- Redis connection errors
- Session/cache not working
- Authentication failures

**Diagnostic Steps:**

```bash
# Test Redis connectivity
redis-cli -h localhost -p 6379 ping
docker exec workspace-qdrant-redis redis-cli ping

# Check Redis logs
docker logs workspace-qdrant-redis --tail=50

# Test from application
docker exec workspace-qdrant-mcp python -c "
import redis
r = redis.Redis(host='redis', port=6379, password='your-password')
print(r.ping())
"
```

**Solutions:**

```bash
# Fix Redis configuration
docker exec workspace-qdrant-redis redis-cli config get requirepass
docker exec workspace-qdrant-redis redis-cli auth your-password

# Clear Redis cache
docker exec workspace-qdrant-redis redis-cli flushall

# Restart Redis
docker-compose restart redis
```

### Data Corruption Issues

**Symptoms:**
- Inconsistent search results
- Vector search errors
- Collection access errors

**Diagnostic Steps:**

```bash
# Check collection status
curl http://localhost:6333/collections
curl http://localhost:6333/collections/your-collection

# Verify data integrity
docker exec workspace-qdrant-mcp python -c "
from workspace_qdrant_mcp.core.client import QdrantClient
client = QdrantClient()
collections = client.get_collections()
print(f'Collections: {collections}')
"

# Check disk space
df -h
docker exec workspace-qdrant-qdrant df -h /qdrant/storage
```

**Solutions:**

```bash
# Backup and restore
docker exec workspace-qdrant-qdrant tar czf /tmp/backup.tar.gz /qdrant/storage
# Stop services, restore from backup, restart

# Reindex data
docker exec workspace-qdrant-mcp workspace-qdrant-admin reindex --collection all

# Recreate collections
curl -X DELETE http://localhost:6333/collections/corrupted-collection
curl -X PUT http://localhost:6333/collections/new-collection -H "Content-Type: application/json" -d '{...}'
```

## Network Issues

### Service Discovery Problems

**Symptoms:**
- Services can't reach each other
- DNS resolution failures
- Connection timeouts

**Diagnostic Steps:**

```bash
# Test DNS resolution
docker exec workspace-qdrant-mcp nslookup qdrant
docker exec workspace-qdrant-mcp nslookup redis
kubectl exec deployment/workspace-qdrant-mcp -n workspace-qdrant-mcp -- nslookup qdrant-service

# Test connectivity
docker exec workspace-qdrant-mcp ping qdrant
docker exec workspace-qdrant-mcp telnet qdrant 6333

# Check network configuration
docker network ls
docker network inspect workspace-qdrant-mcp_workspace-network
kubectl get services -n workspace-qdrant-mcp
```

**Solutions:**

```bash
# Restart networking
docker-compose down && docker-compose up -d

# Fix service names in configuration
# Use service names from docker-compose.yml, not localhost

# Kubernetes - check service endpoints
kubectl get endpoints -n workspace-qdrant-mcp
kubectl describe service qdrant-service -n workspace-qdrant-mcp
```

### Port Conflicts

**Symptoms:**
- Port already in use errors
- Services not accessible
- Connection refused

**Diagnostic Steps:**

```bash
# Check port usage
netstat -tulpn | grep -E "(8000|6333|6379|3000|9090)"
lsof -i :8000

# Check Docker port mappings
docker port workspace-qdrant-mcp
```

**Solutions:**

```bash
# Change ports in .env file
echo "WQM_PORT=8001" >> .env
echo "QDRANT_HTTP_PORT=6334" >> .env

# Kill conflicting processes
sudo kill $(lsof -t -i:8000)

# Use different ports in docker-compose.yml
ports:
  - "8001:8000"  # host:container
```

### Firewall Issues

**Symptoms:**
- External access blocked
- Ingress not working
- Partial connectivity

**Diagnostic Steps:**

```bash
# Check firewall status
sudo ufw status
sudo iptables -L

# Test external connectivity
curl -I http://your-domain.com:8000/health
nmap -p 8000 your-server-ip
```

**Solutions:**

```bash
# Open required ports
sudo ufw allow 8000/tcp
sudo ufw allow 6333/tcp
sudo ufw allow 443/tcp

# Check cloud security groups (AWS example)
aws ec2 describe-security-groups --group-ids sg-xxxxxxxxx
```

## Performance Issues

### High Memory Usage

**Symptoms:**
- Out of memory errors
- Slow response times
- Container restarts

**Diagnostic Steps:**

```bash
# Monitor memory usage
docker stats --no-stream
top -p $(pgrep -f workspace-qdrant-mcp)

# Check memory limits
docker inspect workspace-qdrant-mcp | grep -A 5 "Memory"
kubectl describe pod <pod-name> -n workspace-qdrant-mcp | grep -A 5 "Limits"

# Analyze memory allocation
docker exec workspace-qdrant-mcp python -c "
import psutil
process = psutil.Process()
print(f'Memory: {process.memory_info().rss / 1024 / 1024:.2f} MB')
"
```

**Solutions:**

```bash
# Increase memory limits
# Docker Compose
deploy:
  resources:
    limits:
      memory: 4G

# Kubernetes
resources:
  limits:
    memory: 4Gi

# Optimize application
export WORKSPACE_QDRANT_VECTOR_SIZE=384  # Reduce if possible
export WORKSPACE_QDRANT_MAX_RESULTS=100  # Limit results
```

### High CPU Usage

**Symptoms:**
- High CPU utilization
- Slow response times
- Increased latency

**Diagnostic Steps:**

```bash
# Monitor CPU usage
docker stats --no-stream
htop

# Profile application
docker exec workspace-qdrant-mcp python -c "
import psutil
cpu_percent = psutil.cpu_percent(interval=1)
print(f'CPU Usage: {cpu_percent}%')
"

# Check for CPU-intensive operations
docker logs workspace-qdrant-mcp | grep -i "processing\|indexing\|search"
```

**Solutions:**

```bash
# Scale horizontally
docker-compose up -d --scale workspace-qdrant-mcp=3

# Kubernetes HPA
kubectl get hpa -n workspace-qdrant-mcp

# Optimize queries
# Use filters to reduce search space
# Implement caching for frequent queries
# Batch operations when possible
```

### Disk I/O Issues

**Symptoms:**
- Slow database operations
- High disk utilization
- Timeout errors

**Diagnostic Steps:**

```bash
# Monitor disk I/O
iostat -x 1 5
iotop -o

# Check disk space
df -h
docker system df
kubectl describe pvc -n workspace-qdrant-mcp
```

**Solutions:**

```bash
# Use faster storage
# AWS: gp3 instead of gp2
# Use SSD storage class in Kubernetes

# Optimize Qdrant configuration
# Increase WAL capacity
# Adjust segment size
# Enable compression

# Clean up old data
docker exec workspace-qdrant-mcp workspace-qdrant-admin cleanup --older-than 30d
```

## Security Issues

### Authentication Failures

**Symptoms:**
- Unauthorized errors
- Token validation failures
- Access denied messages

**Diagnostic Steps:**

```bash
# Check API keys
curl -H "api-key: wrong-key" http://localhost:6333/collections
echo $QDRANT_API_KEY

# Test authentication
docker exec workspace-qdrant-mcp python -c "
from workspace_qdrant_mcp.core.client import QdrantClient
client = QdrantClient()
try:
    collections = client.get_collections()
    print('Authentication successful')
except Exception as e:
    print(f'Authentication failed: {e}')
"
```

**Solutions:**

```bash
# Update API keys
kubectl create secret generic workspace-qdrant-mcp-secrets \
  --from-literal=QDRANT_API_KEY="new-secure-key" \
  --dry-run=client -o yaml | kubectl apply -f -

# Restart services to pick up new secrets
kubectl rollout restart deployment/workspace-qdrant-mcp -n workspace-qdrant-mcp
```

### TLS/SSL Issues

**Symptoms:**
- Certificate errors
- HTTPS not working
- Insecure connection warnings

**Diagnostic Steps:**

```bash
# Check certificates
openssl x509 -in docker/nginx/ssl/cert.pem -text -noout
curl -k -I https://localhost/health

# Verify certificate chain
curl -vvI https://your-domain.com/health
```

**Solutions:**

```bash
# Generate new certificates
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout docker/nginx/ssl/key.pem \
  -out docker/nginx/ssl/cert.pem

# Use cert-manager in Kubernetes
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml
```

### Container Vulnerabilities

**Symptoms:**
- Security scan failures
- Known CVEs in base images
- Compliance issues

**Diagnostic Steps:**

```bash
# Scan for vulnerabilities
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy image workspace-qdrant-mcp:latest

# Check base image versions
docker inspect workspace-qdrant-mcp:latest | grep -A 10 "RootFS"
```

**Solutions:**

```bash
# Update base images
# Rebuild with latest base images
docker build --pull -f docker/Dockerfile -t workspace-qdrant-mcp:latest .

# Use distroless images
FROM gcr.io/distroless/python3

# Run as non-root user
USER 65534:65534
```

## Storage Issues

### Persistent Volume Problems

**Symptoms:**
- Data loss after restart
- Mount failures
- Permission denied errors

**Diagnostic Steps:**

```bash
# Check volume mounts
docker inspect workspace-qdrant-mcp | grep -A 10 "Mounts"
kubectl describe pod <pod-name> -n workspace-qdrant-mcp | grep -A 10 "Volumes"

# Check permissions
docker exec workspace-qdrant-mcp ls -la /app/data
kubectl exec deployment/workspace-qdrant-mcp -n workspace-qdrant-mcp -- ls -la /app/data
```

**Solutions:**

```bash
# Fix permissions
docker exec --user root workspace-qdrant-mcp chown -R 65534:65534 /app/data

# Kubernetes PVC issues
kubectl describe pvc workspace-qdrant-mcp-data -n workspace-qdrant-mcp
kubectl get storageclass
```

### Disk Space Issues

**Symptoms:**
- No space left on device
- Write failures
- Container crashes

**Diagnostic Steps:**

```bash
# Check disk usage
df -h
docker system df
docker exec workspace-qdrant-mcp df -h

# Kubernetes
kubectl describe node <node-name>
kubectl top node
```

**Solutions:**

```bash
# Clean up Docker
docker system prune -a -f
docker volume prune -f

# Expand volumes
# Kubernetes
kubectl patch pvc workspace-qdrant-mcp-data -n workspace-qdrant-mcp -p '{"spec":{"resources":{"requests":{"storage":"100Gi"}}}}'

# Clean up old data
docker exec workspace-qdrant-mcp find /app/logs -type f -mtime +7 -delete
```

## Monitoring Issues

### Metrics Not Available

**Symptoms:**
- Prometheus can't scrape metrics
- Grafana shows no data
- Health checks failing

**Diagnostic Steps:**

```bash
# Check metrics endpoints
curl http://localhost:8000/metrics
curl http://localhost:6333/metrics
curl http://localhost:9090/api/v1/targets

# Verify Prometheus configuration
docker exec workspace-qdrant-prometheus cat /etc/prometheus/prometheus.yml
```

**Solutions:**

```bash
# Fix Prometheus configuration
# Ensure correct service names and ports
# Restart Prometheus to reload config

# Check network connectivity
docker exec workspace-qdrant-prometheus wget -qO- http://workspace-qdrant-mcp:8000/metrics
```

### Log Collection Issues

**Symptoms:**
- Logs not showing in Grafana
- Missing log entries
- Log parsing errors

**Diagnostic Steps:**

```bash
# Check log forwarding
docker logs workspace-qdrant-promtail
kubectl logs deployment/promtail -n workspace-qdrant-mcp

# Verify Loki connectivity
curl http://localhost:3100/ready
curl http://localhost:3100/api/v1/labels
```

**Solutions:**

```bash
# Fix Promtail configuration
# Ensure correct log paths
# Check Loki URL configuration

# Restart log collection
docker-compose restart promtail loki
```

## Kubernetes-Specific Issues

### Pod Startup Issues

**Symptoms:**
- Pods stuck in Pending state
- ImagePullBackOff errors
- CrashLoopBackOff

**Diagnostic Steps:**

```bash
# Check pod status
kubectl get pods -n workspace-qdrant-mcp -o wide
kubectl describe pod <pod-name> -n workspace-qdrant-mcp

# Check events
kubectl get events -n workspace-qdrant-mcp --sort-by='.lastTimestamp'

# Check resource constraints
kubectl describe node <node-name>
```

**Solutions:**

```bash
# ImagePullBackOff
kubectl create secret docker-registry registry-secret \
  --docker-server=your-registry.com \
  --docker-username=your-username \
  --docker-password=your-password \
  -n workspace-qdrant-mcp

# Resource constraints
kubectl describe limitrange -n workspace-qdrant-mcp
kubectl edit deployment workspace-qdrant-mcp -n workspace-qdrant-mcp

# CrashLoopBackOff
kubectl logs <pod-name> -n workspace-qdrant-mcp --previous
```

### Service Mesh Issues (Istio)

**Symptoms:**
- Traffic routing failures
- Sidecar injection problems
- mTLS issues

**Diagnostic Steps:**

```bash
# Check sidecar injection
kubectl get pods -n workspace-qdrant-mcp -o jsonpath='{.items[*].spec.containers[*].name}'

# Verify Istio configuration
kubectl get virtualservice,destinationrule,gateway -n workspace-qdrant-mcp
istioctl analyze -n workspace-qdrant-mcp
```

**Solutions:**

```bash
# Enable sidecar injection
kubectl label namespace workspace-qdrant-mcp istio-injection=enabled

# Fix virtual service configuration
kubectl apply -f istio-configs/

# Check proxy status
istioctl proxy-status
```

## Recovery Procedures

### Complete System Recovery

```bash
#!/bin/bash
# Emergency recovery script

echo "Starting emergency recovery..."

# Stop all services
docker-compose -f docker/docker-compose.yml down
kubectl delete namespace workspace-qdrant-mcp --wait

# Clean up resources
docker system prune -a -f
docker volume prune -f

# Restore from backup
BACKUP_DIR="/backup/$(ls /backup | sort -r | head -n1)"
echo "Restoring from: $BACKUP_DIR"

# Recreate namespace and secrets
kubectl create namespace workspace-qdrant-mcp
kubectl create secret generic workspace-qdrant-mcp-secrets \
  --from-env-file="$BACKUP_DIR/.env" \
  -n workspace-qdrant-mcp

# Deploy services
kubectl apply -f docker/k8s/
# or
docker-compose -f docker/docker-compose.yml up -d

# Wait for services to be ready
kubectl wait --for=condition=ready pod -l app=workspace-qdrant-mcp -n workspace-qdrant-mcp --timeout=300s

# Restore data
kubectl exec deployment/qdrant -n workspace-qdrant-mcp -- tar xzf - -C /qdrant/storage < "$BACKUP_DIR/qdrant-storage.tar.gz"

# Verify recovery
curl http://localhost:8000/health
kubectl get pods -n workspace-qdrant-mcp

echo "Recovery completed"
```

### Data Recovery from Corruption

```bash
# Stop services
kubectl scale deployment workspace-qdrant-mcp --replicas=0 -n workspace-qdrant-mcp
kubectl scale deployment qdrant --replicas=0 -n workspace-qdrant-mcp

# Mount volume for data recovery
kubectl run data-recovery --image=busybox --rm -it \
  --overrides='{"spec":{"containers":[{"name":"data-recovery","image":"busybox","command":["/bin/sh"],"stdin":true,"tty":true,"volumeMounts":[{"name":"qdrant-storage","mountPath":"/data"}]}],"volumes":[{"name":"qdrant-storage","persistentVolumeClaim":{"claimName":"qdrant-storage"}}]}}' \
  -n workspace-qdrant-mcp

# Inside the recovery pod:
# Check data integrity
# Restore from backup if needed
# Fix permissions

# Scale services back up
kubectl scale deployment qdrant --replicas=1 -n workspace-qdrant-mcp
kubectl scale deployment workspace-qdrant-mcp --replicas=3 -n workspace-qdrant-mcp
```

### Rolling Back Failed Deployment

```bash
# Kubernetes rollback
kubectl rollout history deployment/workspace-qdrant-mcp -n workspace-qdrant-mcp
kubectl rollout undo deployment/workspace-qdrant-mcp -n workspace-qdrant-mcp
kubectl rollout status deployment/workspace-qdrant-mcp -n workspace-qdrant-mcp

# Docker Compose rollback
docker-compose -f docker/docker-compose.yml pull workspace-qdrant-mcp:previous-version
docker-compose -f docker/docker-compose.yml up -d workspace-qdrant-mcp
```

---

For additional support, check the project's [GitHub Issues](https://github.com/ChrisGVE/workspace-qdrant-mcp/issues) or consult the [Docker Deployment Guide](DOCKER_DEPLOYMENT.md) and [Kubernetes Deployment Guide](KUBERNETES_DEPLOYMENT.md).