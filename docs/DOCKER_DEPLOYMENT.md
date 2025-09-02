# Docker Deployment Guide

This guide provides comprehensive instructions for deploying Workspace Qdrant MCP using Docker and Docker Compose.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Prerequisites](#prerequisites)
3. [Environment Configuration](#environment-configuration)
4. [Docker Compose Deployment](#docker-compose-deployment)
5. [Single Container Deployment](#single-container-deployment)
6. [Container Registry](#container-registry)
7. [Production Deployment](#production-deployment)
8. [Monitoring and Logging](#monitoring-and-logging)
9. [Backup and Recovery](#backup-and-recovery)
10. [Troubleshooting](#troubleshooting)
11. [Security Considerations](#security-considerations)

## Quick Start

### Development Environment

```bash
# Clone the repository
git clone https://github.com/ChrisGVE/workspace-qdrant-mcp.git
cd workspace-qdrant-mcp

# Start development environment
docker-compose -f docker/docker-compose.dev.yml up -d

# Access the application
curl http://localhost:8000/health
```

### Production Environment

```bash
# Set environment variables
export QDRANT_API_KEY="your-secure-api-key"
export REDIS_PASSWORD="your-secure-password"

# Start production environment
docker-compose -f docker/docker-compose.yml -f docker/docker-compose.prod.yml up -d

# Verify deployment
docker-compose ps
```

## Prerequisites

### System Requirements

- **Docker Engine**: 20.10.0 or higher
- **Docker Compose**: 2.0.0 or higher
- **Memory**: Minimum 4GB RAM, recommended 8GB+
- **Storage**: Minimum 50GB free space, recommended 200GB+
- **CPU**: Minimum 2 cores, recommended 4+ cores

### Network Requirements

- **Ports**: 
  - 8000 (Application)
  - 6333, 6334 (Qdrant HTTP/gRPC)
  - 6379 (Redis)
  - 80, 443 (Nginx)
  - 3000 (Grafana)
  - 9090 (Prometheus)

### Software Dependencies

```bash
# Install Docker (Ubuntu/Debian)
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.21.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Verify installation
docker --version
docker-compose --version
```

## Environment Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# Application Configuration
WORKSPACE_QDRANT_HOST=0.0.0.0
WORKSPACE_QDRANT_PORT=8000
WORKSPACE_QDRANT_LOG_LEVEL=INFO
VERSION=latest

# Qdrant Configuration
QDRANT_API_KEY=your-secure-api-key-here
QDRANT_LOG_LEVEL=INFO
QDRANT_HTTP_PORT=6333
QDRANT_GRPC_PORT=6334

# Redis Configuration
REDIS_PASSWORD=your-secure-redis-password
REDIS_MAX_MEMORY=1gb
REDIS_PORT=6379

# Network Configuration
HTTP_PORT=80
HTTPS_PORT=443
WQM_PORT=8000

# Monitoring Configuration
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=your-grafana-password
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
JAEGER_UI_PORT=16686

# Storage Configuration
WORKSPACE_DATA_DIR=./volumes/workspace/data
QDRANT_STORAGE_DIR=./volumes/qdrant/storage
QDRANT_SNAPSHOTS_DIR=./volumes/qdrant/snapshots

# Security Configuration
TLS_CERT_PATH=./ssl/cert.pem
TLS_KEY_PATH=./ssl/key.pem

# Development Configuration (dev environment only)
WORKSPACE_DIR=./workspace
DEBUG_PORT=5678
QDRANT_UI_PORT=8080
REDIS_INSIGHT_PORT=8001
DOCS_PORT=8002
```

### Directory Structure Setup

```bash
# Create necessary directories
mkdir -p volumes/{workspace/{data,logs,tmp},qdrant/{storage,snapshots},redis,prometheus,grafana,loki,jaeger}
mkdir -p ssl backup logs config

# Set proper permissions
sudo chown -R $USER:$USER volumes/
chmod 755 volumes/
chmod 750 volumes/workspace/{data,logs,tmp}
```

### SSL Certificate Setup

For HTTPS support, generate or obtain SSL certificates:

```bash
# Self-signed certificate (development)
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout docker/nginx/ssl/key.pem \
  -out docker/nginx/ssl/cert.pem \
  -subj "/C=US/ST=State/L=City/O=Organization/OU=OrgUnit/CN=localhost"

# Or copy existing certificates
cp /path/to/your/cert.pem docker/nginx/ssl/cert.pem
cp /path/to/your/key.pem docker/nginx/ssl/key.pem
```

## Docker Compose Deployment

### Development Deployment

```bash
# Start all services
docker-compose -f docker/docker-compose.dev.yml up -d

# View logs
docker-compose -f docker/docker-compose.dev.yml logs -f

# Scale application
docker-compose -f docker/docker-compose.dev.yml up -d --scale workspace-qdrant-mcp=3

# Stop services
docker-compose -f docker/docker-compose.dev.yml down

# Stop and remove volumes
docker-compose -f docker/docker-compose.dev.yml down -v
```

### Production Deployment

```bash
# Build and start production services
docker-compose -f docker/docker-compose.yml -f docker/docker-compose.prod.yml up -d --build

# Check service health
docker-compose -f docker/docker-compose.yml -f docker/docker-compose.prod.yml ps

# View logs
docker-compose -f docker/docker-compose.yml -f docker/docker-compose.prod.yml logs -f workspace-qdrant-mcp

# Update services
docker-compose -f docker/docker-compose.yml -f docker/docker-compose.prod.yml pull
docker-compose -f docker/docker-compose.yml -f docker/docker-compose.prod.yml up -d
```

### Service Management

```bash
# Individual service management
docker-compose -f docker/docker-compose.yml restart workspace-qdrant-mcp
docker-compose -f docker/docker-compose.yml stop qdrant
docker-compose -f docker/docker-compose.yml start qdrant

# Resource monitoring
docker stats $(docker-compose -f docker/docker-compose.yml ps -q)

# Execute commands in containers
docker-compose -f docker/docker-compose.yml exec workspace-qdrant-mcp bash
docker-compose -f docker/docker-compose.yml exec qdrant curl http://localhost:6333/health
```

## Single Container Deployment

### Building the Image

```bash
# Build development image
docker build -f docker/Dockerfile --target development -t workspace-qdrant-mcp:dev .

# Build production image
docker build -f docker/Dockerfile --target production -t workspace-qdrant-mcp:prod .

# Build with buildx for multi-platform
docker buildx build --platform linux/amd64,linux/arm64 -f docker/Dockerfile -t workspace-qdrant-mcp:latest .
```

### Running Single Container

```bash
# Run with minimal configuration
docker run -d \
  --name workspace-qdrant-mcp \
  -p 8000:8000 \
  -e QDRANT_HOST=your-qdrant-host \
  -e QDRANT_API_KEY=your-api-key \
  workspace-qdrant-mcp:latest

# Run with volume mounts
docker run -d \
  --name workspace-qdrant-mcp \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/config:/app/config:ro \
  -e QDRANT_HOST=your-qdrant-host \
  -e QDRANT_API_KEY=your-api-key \
  workspace-qdrant-mcp:latest

# Run with custom entrypoint
docker run -it --rm \
  workspace-qdrant-mcp:latest \
  wqm --help
```

### Container Health Monitoring

```bash
# Check container health
docker inspect workspace-qdrant-mcp | jq '.[0].State.Health'

# Follow logs
docker logs -f workspace-qdrant-mcp

# Execute health check manually
docker exec workspace-qdrant-mcp curl http://localhost:8000/health
```

## Container Registry

### Using Pre-built Images

```bash
# Pull from Docker Hub
docker pull chrisgve/workspace-qdrant-mcp:latest

# Pull from GitHub Container Registry
docker pull ghcr.io/chrisgve/workspace-qdrant-mcp:latest

# Pull specific version
docker pull chrisgve/workspace-qdrant-mcp:v0.2.0
```

### Building and Publishing

```bash
# Using the build script
./docker/build-and-push.sh --push --version v0.2.0

# Manual build and push
docker build -f docker/Dockerfile -t chrisgve/workspace-qdrant-mcp:latest .
docker push chrisgve/workspace-qdrant-mcp:latest

# Multi-platform build and push
docker buildx build --platform linux/amd64,linux/arm64 \
  -f docker/Dockerfile \
  -t chrisgve/workspace-qdrant-mcp:latest \
  --push .
```

## Production Deployment

### Pre-deployment Checklist

- [ ] Environment variables configured
- [ ] SSL certificates installed
- [ ] Storage volumes configured
- [ ] Network security configured
- [ ] Monitoring enabled
- [ ] Backup strategy implemented
- [ ] Resource limits set
- [ ] Health checks configured

### Production-specific Configuration

```yaml
# docker-compose.prod.override.yml
version: '3.8'
services:
  workspace-qdrant-mcp:
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"]
      interval: 10s
      timeout: 5s
      retries: 3
      start_period: 30s
```

### Load Balancing

```bash
# Using Docker Swarm
docker swarm init
docker stack deploy -c docker/docker-compose.yml -c docker/docker-compose.prod.yml workspace-qdrant-mcp

# Using external load balancer (Nginx example)
upstream workspace_qdrant_mcp {
    least_conn;
    server localhost:8000 weight=1 max_fails=3 fail_timeout=30s;
    server localhost:8001 weight=1 max_fails=3 fail_timeout=30s;
    server localhost:8002 weight=1 max_fails=3 fail_timeout=30s;
}
```

## Monitoring and Logging

### Accessing Monitoring Services

```bash
# Grafana (default: admin/admin123)
open http://localhost:3000

# Prometheus
open http://localhost:9090

# Jaeger
open http://localhost:16686

# Qdrant dashboard (development)
open http://localhost:8080
```

### Log Management

```bash
# View application logs
docker-compose -f docker/docker-compose.yml logs -f workspace-qdrant-mcp

# View Qdrant logs
docker-compose -f docker/docker-compose.yml logs -f qdrant

# Export logs
docker-compose -f docker/docker-compose.yml logs --no-color > application.log

# Log rotation (production)
echo '{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  }
}' | sudo tee /etc/docker/daemon.json
```

### Metrics Collection

```bash
# Check application metrics
curl http://localhost:8000/metrics

# Check Qdrant metrics
curl http://localhost:6333/metrics

# Custom metrics queries
curl -G 'http://localhost:9090/api/v1/query' \
  --data-urlencode 'query=workspace_qdrant_mcp_requests_total'
```

## Backup and Recovery

### Data Backup

```bash
# Create backup script
cat > backup.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="/backup/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Backup Qdrant data
docker-compose exec -T qdrant tar czf - /qdrant/storage > "$BACKUP_DIR/qdrant-storage.tar.gz"
docker-compose exec -T qdrant tar czf - /qdrant/snapshots > "$BACKUP_DIR/qdrant-snapshots.tar.gz"

# Backup Redis data
docker-compose exec -T redis redis-cli --rdb - > "$BACKUP_DIR/redis.rdb"

# Backup application data
docker-compose exec -T workspace-qdrant-mcp tar czf - /app/data > "$BACKUP_DIR/app-data.tar.gz"

# Backup configuration
cp -r config/ "$BACKUP_DIR/"
cp docker-compose.yml "$BACKUP_DIR/"
cp .env "$BACKUP_DIR/"

echo "Backup completed: $BACKUP_DIR"
EOF

chmod +x backup.sh
```

### Automated Backup

```yaml
# Add to docker-compose.yml
services:
  backup:
    image: alpine:latest
    volumes:
      - qdrant_storage:/qdrant/storage:ro
      - workspace_data:/app/data:ro
      - ./backups:/backup
    command: |
      sh -c '
        apk add --no-cache tar
        while true; do
          BACKUP_DIR="/backup/$$(date +%Y%m%d-%H%M%S)"
          mkdir -p "$$BACKUP_DIR"
          tar czf "$$BACKUP_DIR/qdrant-storage.tar.gz" -C /qdrant/storage .
          tar czf "$$BACKUP_DIR/app-data.tar.gz" -C /app/data .
          find /backup -type d -mtime +7 -exec rm -rf {} +
          sleep 86400  # Daily backup
        done
      '
    restart: unless-stopped
```

### Disaster Recovery

```bash
# Stop services
docker-compose -f docker/docker-compose.yml down

# Restore from backup
RESTORE_DIR="/backup/20231201-120000"
docker-compose exec -T qdrant tar xzf - -C /qdrant/storage < "$RESTORE_DIR/qdrant-storage.tar.gz"
docker-compose exec -T workspace-qdrant-mcp tar xzf - -C /app/data < "$RESTORE_DIR/app-data.tar.gz"

# Restart services
docker-compose -f docker/docker-compose.yml up -d
```

## Troubleshooting

### Common Issues

#### Service Won't Start

```bash
# Check logs
docker-compose logs workspace-qdrant-mcp

# Check container status
docker-compose ps

# Inspect container
docker inspect workspace-qdrant-mcp-workspace-qdrant-mcp-1

# Check resource usage
docker stats
```

#### Network Issues

```bash
# Test network connectivity
docker-compose exec workspace-qdrant-mcp curl http://qdrant:6333/health
docker-compose exec workspace-qdrant-mcp curl http://redis:6379

# Check network configuration
docker network ls
docker network inspect workspace-qdrant-mcp_workspace-network
```

#### Performance Issues

```bash
# Check resource limits
docker-compose exec workspace-qdrant-mcp cat /sys/fs/cgroup/memory/memory.limit_in_bytes

# Monitor resource usage
docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}"

# Check disk usage
docker-compose exec workspace-qdrant-mcp df -h
```

#### Database Issues

```bash
# Check Qdrant status
curl http://localhost:6333/health

# Check collections
curl http://localhost:6333/collections

# Check Redis connection
redis-cli -h localhost -p 6379 ping

# Reset database (destructive)
docker-compose exec qdrant rm -rf /qdrant/storage/*
docker-compose restart qdrant
```

### Debug Mode

```bash
# Start in debug mode
docker-compose -f docker/docker-compose.dev.yml up -d

# Access application shell
docker-compose exec workspace-qdrant-mcp bash

# Run diagnostic commands
docker-compose exec workspace-qdrant-mcp python -m workspace_qdrant_mcp.cli.diagnostics

# Enable debug logging
docker-compose exec workspace-qdrant-mcp \
  env WORKSPACE_QDRANT_LOG_LEVEL=DEBUG \
  python -m workspace_qdrant_mcp.server
```

## Security Considerations

### Container Security

```bash
# Run security scan
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy image workspace-qdrant-mcp:latest

# Check for vulnerabilities
docker run --rm -it \
  anchore/grype workspace-qdrant-mcp:latest
```

### Network Security

- Use private networks for inter-service communication
- Enable TLS encryption for external access
- Implement proper firewall rules
- Use secrets management for sensitive data

### Access Control

```bash
# Create non-root user for containers
echo "USER 1000:1000" >> docker/Dockerfile

# Use read-only root filesystem where possible
docker run --read-only --tmpfs /tmp workspace-qdrant-mcp:latest

# Limit capabilities
docker run --cap-drop=ALL --cap-add=NET_BIND_SERVICE workspace-qdrant-mcp:latest
```

### Secrets Management

```yaml
# Use Docker secrets
secrets:
  qdrant_api_key:
    external: true
  redis_password:
    external: true

services:
  workspace-qdrant-mcp:
    secrets:
      - qdrant_api_key
      - redis_password
    environment:
      QDRANT_API_KEY_FILE: /run/secrets/qdrant_api_key
      REDIS_PASSWORD_FILE: /run/secrets/redis_password
```

---

For additional support and advanced configuration options, refer to the [Kubernetes Deployment Guide](KUBERNETES_DEPLOYMENT.md) and [Container Troubleshooting Guide](CONTAINER_TROUBLESHOOTING.md).