# Production Deployment Guide

## Overview

This guide covers production deployment procedures for workspace-qdrant-mcp, including service installation, Docker orchestration, monitoring setup, and operational procedures validated by Task 90 comprehensive testing.

## Quick Start

```bash
# 1. Validate production readiness
python tests/production_readiness_validator.py

# 2. Deploy with Docker Compose
docker-compose -f docker/docker-compose.yml up -d

# 3. Verify deployment
curl http://localhost:8000/health
curl http://localhost:9090/metrics
```

## Service Installation

### Cross-Platform Service Installation

The workspace-qdrant-mcp daemon can be installed as a system service on Linux, macOS, and Windows.

#### Prerequisites

```bash
# Build the daemon binary
cd rust-engine
cargo build --release --bin memexd-priority
```

#### Linux (systemd)

```bash
# Install as system service
wqm service install --auto-start --system

# User service installation
wqm service install --auto-start --user

# Service management
sudo systemctl start memexd
sudo systemctl stop memexd
sudo systemctl status memexd
sudo systemctl logs memexd
```

#### macOS (launchd)

```bash
# Install as system daemon
wqm service install --auto-start

# User agent installation
wqm service install --auto-start --user

# Service management
launchctl start com.workspace-qdrant-mcp.memexd
launchctl stop com.workspace-qdrant-mcp.memexd
wqm service status
wqm service logs
```

#### Windows

```bash
# Windows service installation (when available)
wqm service install --auto-start
```

### Service Configuration

Service configuration is managed through environment variables and configuration files:

```bash
# Configuration file (optional)
wqm service install --config /path/to/config.json

# Log level configuration
wqm service install --log-level debug

# Service restart
wqm service restart

# Service uninstallation
wqm service uninstall
```

## Docker Deployment

### Production Stack

The production deployment includes:

- **workspace-qdrant-mcp**: Main application service
- **qdrant**: Vector database
- **redis**: Caching and session storage
- **nginx**: Reverse proxy with SSL termination
- **prometheus**: Metrics collection
- **grafana**: Visualization and monitoring
- **jaeger**: Distributed tracing
- **loki**: Log aggregation

### Environment Configuration

Create environment file:

```bash
# Copy environment template
cp docker/.env.example docker/.env

# Edit configuration
vim docker/.env
```

Key environment variables:

```bash
# Application
WQM_PORT=8000
LOG_LEVEL=INFO
WORKSPACE_DIR=/path/to/workspace

# Database
QDRANT_API_KEY=your-secure-key
REDIS_PASSWORD=your-redis-password

# Monitoring
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
GRAFANA_ADMIN_PASSWORD=secure-password

# Security
SSL_CERT_PATH=/path/to/cert.pem
SSL_KEY_PATH=/path/to/key.pem
```

### Deployment Commands

```bash
# Production deployment
docker-compose -f docker/docker-compose.yml up -d

# Scale services
docker-compose -f docker/docker-compose.yml up -d --scale workspace-qdrant-mcp=3

# View logs
docker-compose -f docker/docker-compose.yml logs -f

# Update services
docker-compose -f docker/docker-compose.yml pull
docker-compose -f docker/docker-compose.yml up -d

# Backup data
docker run --rm -v workspace_data:/data -v $(pwd)/backups:/backup alpine tar czf /backup/workspace-$(date +%Y%m%d).tar.gz -C /data .

# Shutdown
docker-compose -f docker/docker-compose.yml down
```

### Health Checks

All services include health checks for monitoring and orchestration:

```bash
# Check service health
curl http://localhost:8000/health
curl http://localhost:8000/health/detailed
curl http://localhost:8000/diagnostics

# Prometheus metrics
curl http://localhost:8000/metrics
curl http://localhost:9090/api/v1/query?query=up

# Container health
docker-compose -f docker/docker-compose.yml ps
```

## Monitoring and Observability

### Prometheus Metrics

Access Prometheus at `http://localhost:9090`

Key metrics to monitor:

- `requests_total` - Request counts by endpoint and status
- `operation_duration_seconds` - Operation latency
- `memory_usage_bytes` - Memory utilization
- `active_connections` - Connection count
- `documents_processed_total` - Document ingestion rate

### Grafana Dashboards

Access Grafana at `http://localhost:3000` (admin/admin123)

Pre-configured dashboards:

- **System Overview** - Resource utilization and health
- **Application Metrics** - Request rates and response times
- **Database Performance** - Qdrant and Redis metrics
- **Error Tracking** - Error rates and types

### Log Aggregation

Structured logs are aggregated through Loki and viewable in Grafana.

Log levels and components:

```json
{
  "level": "info",
  "message": "Request completed",
  "component": "server",
  "operation": "search",
  "duration": 0.234,
  "status_code": 200
}
```

### Alerting

Prometheus alerts are configured for critical conditions:

- High error rate (>5% for 5 minutes)
- High memory usage (>90% for 10 minutes)
- High CPU usage (>95% for 5 minutes)
- Service unavailable (down for 1 minute)
- Disk space low (<10% free)

## Security Configuration

### Container Security

All containers implement security best practices:

```yaml
security_opt:
  - no-new-privileges:true
cap_drop:
  - ALL
cap_add:
  - CHOWN
  - DAC_OVERRIDE
read_only: false
tmpfs:
  - /tmp:exec,nodev,nosuid,size=100m
```

### Network Security

Services are isolated in separate networks:

- `workspace-network` - Application services
- `monitoring` - Monitoring stack

### Secrets Management

Sensitive configuration uses environment variable substitution:

```yaml
environment:
  - QDRANT_API_KEY=${QDRANT_API_KEY}
  - REDIS_PASSWORD=${REDIS_PASSWORD}
```

### SSL/TLS Configuration

NGINX provides SSL termination:

```bash
# Generate SSL certificates (development)
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout docker/nginx/ssl/key.pem \
  -out docker/nginx/ssl/cert.pem

# Production: Use Let's Encrypt or commercial certificates
```

## Backup and Recovery

### Data Backup

Critical data volumes require regular backup:

```bash
# Backup workspace data
docker run --rm -v workspace_data:/data -v $(pwd)/backups:/backup alpine \
  tar czf /backup/workspace-$(date +%Y%m%d).tar.gz -C /data .

# Backup Qdrant data
docker run --rm -v qdrant_storage:/data -v $(pwd)/backups:/backup alpine \
  tar czf /backup/qdrant-$(date +%Y%m%d).tar.gz -C /data .

# Backup Redis data
docker run --rm -v redis_data:/data -v $(pwd)/backups:/backup alpine \
  tar czf /backup/redis-$(date +%Y%m%d).tar.gz -C /data .
```

### Configuration Backup

```bash
# Backup configuration
tar czf config-backup-$(date +%Y%m%d).tar.gz docker/config/ docker/.env
```

### Recovery Procedures

```bash
# Stop services
docker-compose -f docker/docker-compose.yml down

# Restore data
docker run --rm -v workspace_data:/data -v $(pwd)/backups:/backup alpine \
  tar xzf /backup/workspace-20240101.tar.gz -C /data

# Start services
docker-compose -f docker/docker-compose.yml up -d
```

### Automated Backup

Create automated backup script:

```bash
#!/bin/bash
# backup.sh - Automated backup script

BACKUP_DIR="/backups/workspace-qdrant-mcp"
DATE=$(date +%Y%m%d-%H%M%S)

mkdir -p $BACKUP_DIR

# Backup data volumes
docker run --rm -v workspace_data:/data -v $BACKUP_DIR:/backup alpine \
  tar czf /backup/workspace-$DATE.tar.gz -C /data .

docker run --rm -v qdrant_storage:/data -v $BACKUP_DIR:/backup alpine \
  tar czf /backup/qdrant-$DATE.tar.gz -C /data .

# Cleanup old backups (keep 7 days)
find $BACKUP_DIR -name "*.tar.gz" -mtime +7 -delete

# Backup to remote storage (optional)
# aws s3 sync $BACKUP_DIR s3://your-backup-bucket/workspace-qdrant-mcp/
```

## Update and Upgrade Procedures

### Zero-Downtime Updates

For zero-downtime updates with multiple instances:

```bash
# Rolling update script
#!/bin/bash
# rolling-update.sh

# Pull new images
docker-compose -f docker/docker-compose.yml pull

# Update services one by one
for service in workspace-qdrant-mcp-1 workspace-qdrant-mcp-2 workspace-qdrant-mcp-3; do
  echo "Updating $service..."
  docker-compose -f docker/docker-compose.yml up -d --no-deps $service
  
  # Wait for health check
  sleep 30
  
  # Verify health
  curl -f http://localhost:8000/health || {
    echo "Health check failed for $service"
    exit 1
  }
done

echo "Rolling update completed successfully"
```

### Configuration Updates

```bash
# Update configuration
vim docker/.env

# Restart affected services
docker-compose -f docker/docker-compose.yml restart workspace-qdrant-mcp

# Verify configuration
curl http://localhost:8000/diagnostics
```

### Database Migrations

```bash
# Backup before migration
./scripts/backup.sh

# Run migrations (if applicable)
docker-compose -f docker/docker-compose.yml exec workspace-qdrant-mcp \
  python -m workspace_qdrant_mcp.migrations.run

# Verify migration
curl http://localhost:8000/health/detailed
```

## Performance Tuning

### Resource Allocation

Adjust resource limits in docker-compose.yml:

```yaml
deploy:
  resources:
    limits:
      cpus: '4.0'
      memory: 4G
    reservations:
      cpus: '1.0'
      memory: 1G
```

### Database Tuning

#### Qdrant Configuration

```yaml
environment:
  - QDRANT__STORAGE__WAL_CAPACITY_MB=64
  - QDRANT__STORAGE__PERFORMANCE__MAX_SEARCH_THREADS=4
  - QDRANT__STORAGE__PERFORMANCE__MAX_OPTIMIZATION_THREADS=2
```

#### Redis Configuration

```yaml
command: >
  redis-server
  --maxmemory 1gb
  --maxmemory-policy allkeys-lru
  --save 900 1
  --save 300 10
```

### Application Tuning

```yaml
environment:
  - WORKSPACE_QDRANT_WORKERS=4
  - WORKSPACE_QDRANT_MAX_CONNECTIONS=100
  - WORKSPACE_QDRANT_TIMEOUT=30
```

## Troubleshooting

### Common Issues

#### Service Won't Start

```bash
# Check logs
docker-compose -f docker/docker-compose.yml logs workspace-qdrant-mcp

# Check configuration
docker-compose -f docker/docker-compose.yml config

# Check health
curl -v http://localhost:8000/health
```

#### High Memory Usage

```bash
# Check memory usage
docker stats

# Check application metrics
curl http://localhost:8000/metrics | grep memory

# Restart if necessary
docker-compose -f docker/docker-compose.yml restart workspace-qdrant-mcp
```

#### Database Connection Issues

```bash
# Check Qdrant health
curl http://localhost:6333/health

# Check Redis connection
docker-compose -f docker/docker-compose.yml exec redis redis-cli ping

# Check network connectivity
docker network ls
docker network inspect docker_workspace-network
```

### Log Analysis

```bash
# Application logs
docker-compose -f docker/docker-compose.yml logs -f workspace-qdrant-mcp

# Error logs only
docker-compose -f docker/docker-compose.yml logs workspace-qdrant-mcp 2>&1 | grep -i error

# Structured log querying (if using Loki)
curl -G -s "http://localhost:3100/loki/api/v1/query" \
  --data-urlencode 'query={service="workspace-qdrant-mcp"} |= "error"' \
  --data-urlencode 'limit=100'
```

## Validation and Testing

### Production Readiness Validation

Run the comprehensive validation script:

```bash
# Full validation
python tests/production_readiness_validator.py

# JSON output
python tests/production_readiness_validator.py --format json

# Strict validation (for CI/CD)
python tests/production_readiness_validator.py --strict --output validation-report.json
```

### Test Suite Execution

```bash
# Production deployment tests
pytest tests/test_production_deployment.py -v

# Docker deployment tests
pytest tests/test_docker_deployment.py -v

# Monitoring integration tests
pytest tests/test_monitoring_integration.py -v

# Load testing
pytest tests/test_production_deployment.py::TestProductionReadiness::test_resource_monitoring -v
```

### Health Check Validation

```bash
# Basic health check
curl http://localhost:8000/health

# Detailed health with components
curl http://localhost:8000/health/detailed | jq

# System diagnostics
curl http://localhost:8000/diagnostics | jq '.system_info'

# Metrics export
curl http://localhost:8000/metrics | head -20
```

## Maintenance Procedures

### Regular Maintenance Tasks

#### Daily
- Monitor service health and performance metrics
- Check log files for errors or warnings
- Verify backup completion

#### Weekly  
- Review resource utilization trends
- Check for available updates
- Clean up old log files and temporary data

#### Monthly
- Review and rotate access logs
- Update security certificates (if applicable)
- Perform backup restore testing
- Review alerting thresholds and rules

### Monitoring Checklist

- [ ] All services healthy (`docker-compose ps`)
- [ ] Prometheus targets up (`http://localhost:9090/targets`)
- [ ] No critical alerts (`http://localhost:9090/alerts`)
- [ ] Log aggregation working (`http://localhost:3000`)
- [ ] Backup jobs successful
- [ ] SSL certificates valid
- [ ] Resource utilization within thresholds

## Production Deployment Checklist

### Pre-Deployment

- [ ] All tests passing (`pytest tests/test_production_deployment.py`)
- [ ] Production readiness validation passed
- [ ] Configuration reviewed and validated
- [ ] Backup procedures tested
- [ ] Rollback plan documented
- [ ] Monitoring alerts configured
- [ ] SSL certificates valid
- [ ] Resource limits appropriate

### Deployment

- [ ] Environment variables configured
- [ ] Docker images built and tested
- [ ] Services deployed (`docker-compose up -d`)
- [ ] Health checks passing
- [ ] Monitoring data flowing
- [ ] Load balancer configured (if applicable)
- [ ] DNS records updated (if applicable)

### Post-Deployment

- [ ] All services healthy and responding
- [ ] Metrics collection working
- [ ] Log aggregation operational
- [ ] Backup jobs scheduled and tested
- [ ] Alerts configured and tested
- [ ] Documentation updated
- [ ] Team notified of deployment
- [ ] Performance baseline established

## Support and Resources

### Documentation
- [API Documentation](docs/api/)
- [Configuration Reference](docs/configuration.md)
- [Monitoring Guide](monitoring/README.md)
- [Security Guidelines](docs/security.md)

### Monitoring URLs
- Application: http://localhost:8000
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000
- Jaeger: http://localhost:16686

### Emergency Contacts
- Primary: [Your contact information]
- Secondary: [Backup contact information]
- Escalation: [Management contact]

### Runbooks
- [Service Recovery Procedures](docs/runbooks/service-recovery.md)
- [Database Recovery](docs/runbooks/database-recovery.md)
- [Performance Issues](docs/runbooks/performance-issues.md)
- [Security Incidents](docs/runbooks/security-incidents.md)

---

This guide provides comprehensive procedures for production deployment of workspace-qdrant-mcp. For additional support, consult the documentation links or contact the development team.