# Observability Guide

## Overview

Workspace Qdrant MCP provides comprehensive observability through structured logging, metrics collection, and health monitoring. This guide covers setup, configuration, and usage of the observability features.

## Structured Logging

### Configuration

The system uses structured JSON logging by default. Logging can be configured via:

1. **Configuration file**: `config/logging.conf`
2. **Environment variables**: `LOG_LEVEL`, `LOG_FORMAT`, `LOG_FILE`
3. **Programmatic configuration**

### Log Levels

- **DEBUG**: Detailed diagnostic information
- **INFO**: General operational events
- **WARNING**: Potentially problematic situations
- **ERROR**: Error events that don't stop operation
- **CRITICAL**: Serious error events that may abort operation

### Log Format

All logs are formatted as JSON with standardized fields:

```json
{
  "timestamp": "2025-01-02T15:30:45",
  "level": "INFO",
  "logger": "workspace_qdrant_mcp.core.client",
  "module": "client",
  "function": "initialize",
  "line": 125,
  "message": "Client initialized",
  "project": "my-project",
  "collections": ["my-project", "my-project.frontend"]
}
```

### Log Rotation

Automated log rotation is configured via logrotate:

- **Production**: Daily rotation, 30 days retention
- **Development**: Size-based rotation (100MB), 5 files retention
- **Compression**: Enabled for older log files

## Metrics Collection

### Prometheus Integration

The system exposes metrics in Prometheus format at `/metrics` endpoint:

```bash
# Start metrics server
wqm observability start-metrics-server --port 8080

# View metrics
curl http://localhost:8080/metrics
```

### Key Metrics

- **System Health**: `system_health_status`
- **Request Metrics**: `mcp_requests_total`, `mcp_request_duration_seconds`
- **Search Operations**: `search_operations_active`, `search_duration_seconds`
- **Document Processing**: `documents_processed_total`, `document_processing_queue_length`
- **Qdrant Operations**: `qdrant_operations_total`, `qdrant_collections_count`
- **File Watchers**: `file_watchers_active`, `file_watchers_errors`

## Health Monitoring

### Health Checks

The system performs comprehensive health checks:

1. **System Resources**: Memory, CPU, disk usage
2. **Qdrant Connectivity**: Database connection and collection health
3. **Embedding Service**: Model availability and performance
4. **File Watchers**: Watch configuration and operation status
5. **Configuration**: Settings validation

### Health Endpoints

```bash
# Basic health check
curl http://localhost:8080/health

# Detailed diagnostics
curl http://localhost:8080/health/detailed

# Individual component check
curl http://localhost:8080/health/qdrant
```

### CLI Health Commands

```bash
# System status overview
wqm observability health

# Detailed component diagnostics
wqm observability diagnostics

# Start health monitoring
wqm observability monitor --interval 30
```

## Grafana Dashboard

### Installation

1. Import the dashboard: `monitoring/grafana/workspace-qdrant-mcp-dashboard.json`
2. Configure Prometheus data source
3. Set refresh interval to 30 seconds

### Dashboard Panels

- **System Health Status**: Overall health indicator
- **Request Rate**: Incoming request volume
- **Response Time Percentiles**: Performance metrics (50th, 95th, 99th)
- **Error Rate**: Request failure percentage
- **Active Searches**: Concurrent search operations
- **Document Processing Queue**: Processing backlog and rate
- **Memory Usage**: Process memory consumption
- **Qdrant Collections**: Collection count and health
- **File Watchers Status**: Watch system health
- **Health Check Response Times**: Component check performance

## Production Setup

### Docker Deployment

```yaml
version: '3.8'
services:
  workspace-qdrant-mcp:
    image: workspace-qdrant-mcp:latest
    environment:
      - LOG_LEVEL=INFO
      - METRICS_ENABLED=true
      - HEALTH_CHECK_ENABLED=true
    volumes:
      - ./logs:/app/logs
      - ./config/logging.conf:/app/config/logging.conf
    ports:
      - "8080:8080"  # Metrics and health endpoints
```

### Kubernetes Deployment

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: logging-config
data:
  logging.conf: |
    # Logging configuration content
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: workspace-qdrant-mcp
spec:
  template:
    spec:
      containers:
      - name: app
        image: workspace-qdrant-mcp:latest
        env:
        - name: LOG_LEVEL
          value: "INFO"
        - name: METRICS_ENABLED
          value: "true"
        volumeMounts:
        - name: logging-config
          mountPath: /app/config/logging.conf
          subPath: logging.conf
      volumes:
      - name: logging-config
        configMap:
          name: logging-config
```

### Log Aggregation

#### ELK Stack Integration

```json
{
  "input": {
    "beats": {
      "port": 5044
    }
  },
  "filter": {
    "json": {
      "source": "message"
    }
  },
  "output": {
    "elasticsearch": {
      "hosts": ["elasticsearch:9200"]
    }
  }
}
```

#### Fluentd Configuration

```ruby
<source>
  @type tail
  path /app/logs/*.log
  pos_file /var/log/fluentd/workspace-qdrant-mcp.log.pos
  tag workspace.qdrant.mcp
  format json
</source>

<match workspace.qdrant.mcp>
  @type elasticsearch
  host elasticsearch
  port 9200
  index_name workspace-qdrant-mcp
</match>
```

## Alerting

### Prometheus Alerting Rules

```yaml
groups:
  - name: workspace-qdrant-mcp.rules
    rules:
    - alert: SystemUnhealthy
      expr: system_health_status == 0
      for: 5m
      annotations:
        summary: "Workspace Qdrant MCP system is unhealthy"
    
    - alert: HighErrorRate
      expr: rate(mcp_requests_total{status="error"}[5m]) / rate(mcp_requests_total[5m]) > 0.05
      for: 2m
      annotations:
        summary: "High error rate detected"
    
    - alert: SlowResponses
      expr: histogram_quantile(0.95, mcp_request_duration_seconds_bucket) > 2
      for: 5m
      annotations:
        summary: "95th percentile response time is high"
```

## Troubleshooting

### Common Issues

1. **Missing Log Files**: Check permissions and directory existence
2. **Metrics Not Available**: Ensure metrics server is started
3. **Health Checks Failing**: Verify component connectivity
4. **High Memory Usage**: Review log retention settings

### Debug Commands

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
wqm admin status

# Check log file permissions
ls -la logs/

# Test health endpoint
curl -v http://localhost:8080/health

# Validate metrics format
curl http://localhost:8080/metrics | head -20
```

### Log Analysis

```bash
# Search for errors
grep -E '"level":"ERROR"' logs/workspace-qdrant-mcp.log

# Extract performance metrics
jq -r 'select(.message | contains("duration")) | .message' logs/workspace-qdrant-mcp.log

# Count requests by type
jq -r '.tool_name' logs/workspace-qdrant-mcp.log | sort | uniq -c
```

## Best Practices

1. **Log Retention**: Configure appropriate retention periods
2. **Metrics Storage**: Use time-series database for metrics
3. **Alert Tuning**: Avoid alert fatigue with proper thresholds
4. **Regular Monitoring**: Review dashboards and alerts regularly
5. **Performance Impact**: Monitor observability overhead
6. **Security**: Protect sensitive data in logs
7. **Documentation**: Keep observability docs updated

## Integration Examples

### Custom Metrics

```python
from workspace_qdrant_mcp.observability import metrics_instance

# Record custom business metric
metrics_instance.increment_counter(
    "documents_analyzed", 
    source="pdf",
    project="my-project"
)

# Track operation duration
with metrics_instance.timer("custom_operation_duration"):
    # Your operation here
    pass
```

### Custom Health Checks

```python
from workspace_qdrant_mcp.observability import health_checker_instance

async def check_external_service():
    # Your health check logic
    return {
        "status": "healthy",
        "message": "Service operational",
        "details": {"response_time": 0.1}
    }

health_checker_instance.register_check(
    "external_service",
    check_external_service,
    timeout_seconds=10.0
)
```

This observability system provides comprehensive monitoring capabilities for production deployments of Workspace Qdrant MCP.