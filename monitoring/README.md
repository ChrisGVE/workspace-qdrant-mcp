# Workspace Qdrant MCP Monitoring Setup

This directory contains comprehensive monitoring, alerting, and observability configurations for the Workspace Qdrant MCP service. The monitoring stack provides production-ready observability with multiple deployment options and detailed operational runbooks.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                        │
├─────────────────┬─────────────────┬─────────────────────────┤
│   Python MCP    │   Rust Engine   │    Qdrant Vector DB    │
│    Server       │                 │                         │
│  (Port 8000)    │  (Port 8002)    │     (Port 6333)        │
└─────────────────┴─────────────────┴─────────────────────────┘
         │                 │                      │
         └─────────────────┼──────────────────────┘
                           │
┌─────────────────────────────────────────────────────────────┐
│                 Metrics Collection                          │
├─────────────────┬─────────────────┬─────────────────────────┤
│   Prometheus    │  Health Checks  │    Node Exporter       │
│  (Port 9090)    │  (Port 8080)    │    (Port 9100)         │
└─────────────────┴─────────────────┴─────────────────────────┘
         │                 │                      │
         └─────────────────┼──────────────────────┘
                           │
┌─────────────────────────────────────────────────────────────┐
│              Alerting and Visualization                     │
├─────────────────┬─────────────────┬─────────────────────────┤
│  Alertmanager   │    Grafana      │    Log Aggregation     │
│  (Port 9093)    │  (Port 3000)    │   (ELK/Loki Stack)     │
└─────────────────┴─────────────────┴─────────────────────────┘
```

## Quick Start

### 1. Choose Your Monitoring Stack

#### Option A: Full Prometheus + Grafana + ELK Stack (Recommended for Production)
```bash
# Start Prometheus monitoring
docker-compose -f monitoring/prometheus/docker-compose.yml up -d

# Start Grafana dashboards
docker-compose -f monitoring/grafana/docker-compose.yml up -d

# Start ELK log aggregation
docker-compose -f monitoring/log-aggregation/elasticsearch-logstash-kibana.yml up -d
```

#### Option B: Lightweight Loki Stack (Recommended for Development)
```bash
# Start Loki log aggregation with Grafana
docker-compose -f monitoring/log-aggregation/loki-promtail-grafana.yml up -d

# Start Prometheus for metrics
docker-compose -f monitoring/prometheus/docker-compose.yml up -d
```

### 2. Start Health Check Server
```bash
# Install dependencies
pip install aiohttp psutil

# Run health check server
python monitoring/health-checks/health_check_server.py
```

### 3. Access Monitoring Interfaces

| Service | URL | Purpose |
|---------|-----|---------|
| Grafana | http://localhost:3000 | Dashboards and visualization |
| Prometheus | http://localhost:9090 | Metrics query interface |
| Alertmanager | http://localhost:9093 | Alert management |
| Kibana | http://localhost:5601 | Log search and analysis |
| Health Checks | http://localhost:8080/health | Service health monitoring |

Default credentials:
- Grafana: admin/admin
- Kibana: elastic/elastic

## Components Overview

### 📊 Metrics Collection

#### Prometheus
- **Configuration**: `prometheus/prometheus.yml`
- **Scrapes**: Application metrics, system metrics, health checks
- **Retention**: 15 days (configurable)
- **Storage**: Local filesystem (production should use remote storage)

#### Application Metrics Endpoints
```bash
# Main application metrics
curl http://localhost:8000/metrics

# Health check metrics
curl http://localhost:8080/metrics

# Qdrant vector database metrics  
curl http://localhost:6333/metrics
```

#### Key Metrics Collected
- HTTP request rate, latency, and error rate
- Vector search performance metrics
- Document ingestion rates and errors
- Memory, CPU, and disk usage
- Database connection pool status
- Queue sizes and processing times

### 🚨 Alerting Rules

#### Critical Alerts (Immediate Response Required)
- **HighErrorRate**: Error rate > 5% for 2+ minutes
- **ServiceDown**: Service unreachable for 30+ seconds
- **DatabaseConnectionFailures**: DB connection failures > 0.1/sec
- **HighCPUUsage**: CPU usage > 90% for 5+ minutes
- **LowDiskSpace**: Available disk space < 10%

#### Warning Alerts (Monitor and Plan Response)
- **SlowResponseTime**: 95th percentile > 2 seconds
- **HighMemoryUsage**: Memory usage > 80%
- **QueueBacklog**: Task queue > 1000 items
- **HighVectorSearchLatency**: Search latency > 5 seconds

### 📈 Dashboards

#### Application Overview Dashboard
- Request rate and error rate
- Response time percentiles
- Memory and CPU usage
- Vector search performance
- Document ingestion metrics

#### System Metrics Dashboard  
- CPU usage by core
- Memory breakdown (used, cached, buffers)
- Disk I/O and filesystem usage
- Network I/O and errors
- Load average and process counts

#### Custom Business Metrics Dashboard
- User activity patterns
- Search query analytics
- Document collection statistics
- Performance trends over time

### 📋 Log Aggregation

#### ELK Stack (Elasticsearch + Logstash + Kibana)
- **Best for**: Large scale, complex log analysis
- **Features**: Full-text search, advanced analytics
- **Resource usage**: High (4GB+ RAM recommended)

#### Loki Stack (Loki + Promtail + Grafana)
- **Best for**: Cloud-native, cost-effective logging
- **Features**: Label-based indexing, integrates with Grafana
- **Resource usage**: Low (1GB+ RAM sufficient)

#### Log Parsing Features
- Structured JSON log parsing
- Vector search performance extraction
- Error categorization and stack trace parsing
- HTTP request log analysis
- Automated log rotation and retention

### 🔧 Health Checks

#### Health Check Endpoints
```bash
# Comprehensive health check
curl http://localhost:8080/health

# Quick health summary  
curl http://localhost:8080/health/summary

# Kubernetes readiness probe
curl http://localhost:8080/ready

# Kubernetes liveness probe
curl http://localhost:8080/alive

# Prometheus metrics from health checks
curl http://localhost:8080/metrics
```

#### Health Check Components
- Database connectivity verification
- Memory and disk usage monitoring
- CPU usage assessment
- Application endpoint testing
- External dependency health verification

## Configuration

### Environment Variables
```bash
# Required for full monitoring setup
export PROMETHEUS_DATA_PATH=/opt/prometheus/data
export GRAFANA_DATA_PATH=/opt/grafana/data
export ELASTICSEARCH_DATA_PATH=/opt/elasticsearch/data

# Optional: Alert notification settings
export SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
export EMAIL_SMTP_SERVER=smtp.company.com
export PAGERDUTY_SERVICE_KEY=your-pagerduty-key
```

### Prometheus Configuration Customization
```yaml
# monitoring/prometheus/prometheus.yml
global:
  scrape_interval: 15s        # Adjust based on load
  evaluation_interval: 15s    # How often to evaluate alerts

scrape_configs:
  - job_name: 'workspace-qdrant-mcp'
    static_configs:
      - targets: ['localhost:8000']  # Update for your deployment
    scrape_interval: 15s
```

### Alert Threshold Tuning
```yaml
# monitoring/prometheus/rules/application.yml
- alert: HighErrorRate
  expr: |
    rate(http_requests_total{status=~"5.."}[5m]) / 
    rate(http_requests_total[5m]) * 100 > 5  # Adjust threshold
  for: 2m  # Adjust evaluation period
```

## Deployment Options

### Local Development
```bash
# Use lightweight stack
docker-compose -f monitoring/log-aggregation/loki-promtail-grafana.yml up -d
python monitoring/health-checks/health_check_server.py
```

### Staging Environment
```bash
# Use full stack with reduced retention
docker-compose -f monitoring/prometheus/docker-compose.yml up -d
docker-compose -f monitoring/log-aggregation/loki-promtail-grafana.yml up -d
```

### Production Environment
```bash
# Use full stack with remote storage and HA setup
# See production-deployment.md for detailed instructions
```

### Kubernetes Deployment
```bash
# Apply Kubernetes manifests
kubectl apply -f monitoring/kubernetes/
kubectl apply -f monitoring/kubernetes/prometheus/
kubectl apply -f monitoring/kubernetes/grafana/
```

## Operational Runbooks

Detailed runbooks are available for common incidents:

- **[High Error Rate](runbooks/high-error-rate.md)**: Comprehensive guide for handling elevated error rates
- **[Service Down](runbooks/service-down.md)**: Emergency procedures for service outages
- **[Database Issues](runbooks/db-connection-failures.md)**: Database connectivity problems
- **[Performance Issues](runbooks/slow-response.md)**: Response time degradation
- **[Resource Issues](runbooks/high-memory.md)**: Memory and CPU problems

Each runbook includes:
- Immediate response steps (first 5 minutes)
- Investigation procedures
- Common causes and solutions
- Recovery actions
- Post-incident tasks
- Communication templates

## SLA and SLO Monitoring

### Service Level Objectives (SLOs)
- **Availability**: 99.9% uptime (< 8.76 hours downtime/year)
- **Error Rate**: < 1% of requests result in 5xx errors
- **Response Time**: 95th percentile < 1 second
- **Search Latency**: 95th percentile vector search < 500ms

### SLA Tracking Queries
```promql
# Availability SLO
(
  sum(rate(up[5m])) /
  count(up)
) * 100

# Error rate SLO
(
  sum(rate(http_requests_total{status=~"5.."}[5m])) /
  sum(rate(http_requests_total[5m]))
) * 100

# Response time SLO
histogram_quantile(0.95, 
  rate(http_request_duration_seconds_bucket[5m])
)
```

## Security Considerations

### Access Control
```yaml
# Grafana security settings
auth:
  disable_login_form: false
  disable_signout_menu: false
security:
  admin_user: admin
  admin_password: ${GRAFANA_ADMIN_PASSWORD}
```

### Network Security
- Use firewall rules to restrict access to monitoring ports
- Enable HTTPS/TLS for all monitoring interfaces
- Implement authentication for Prometheus and Alertmanager
- Use secrets management for sensitive configuration

### Data Privacy
- Scrub sensitive data from logs before indexing
- Implement log retention policies
- Use field-level encryption for sensitive metrics
- Regular security audits of monitoring infrastructure

## Troubleshooting

### Common Issues

#### High Resource Usage
```bash
# Check Prometheus disk usage
du -sh /opt/prometheus/data

# Reduce retention period
prometheus --storage.tsdb.retention.time=7d

# Check Elasticsearch disk usage  
curl -X GET "localhost:9200/_cat/indices?v"
```

#### Missing Metrics
```bash
# Check Prometheus targets
curl http://localhost:9090/api/v1/targets

# Verify service is exposing metrics
curl http://localhost:8000/metrics | head -20

# Check firewall rules
netstat -tulpn | grep :8000
```

#### Alert Fatigue
- Review and tune alert thresholds
- Implement alert grouping and inhibition rules
- Use different notification channels for different severities
- Regular review of alert effectiveness

### Performance Optimization

#### Prometheus Optimization
```yaml
# Reduce memory usage
global:
  scrape_interval: 30s        # Increase interval
  evaluation_interval: 30s

# Limit metrics retention
storage:
  tsdb:
    retention.time: 7d        # Reduce retention
    retention.size: 10GB      # Set size limit
```

#### Log Aggregation Optimization
```yaml
# Reduce log volume
input:
  beats:
    port: 5044
filter:
  drop:
    if: '[level] == "DEBUG"'  # Drop debug logs in production
```

## Support and Maintenance

### Regular Maintenance Tasks
- [ ] Weekly: Review alert effectiveness and tune thresholds
- [ ] Monthly: Clean up old metrics and logs
- [ ] Quarterly: Review and update dashboards
- [ ] Annually: Review SLA/SLO targets and monitoring strategy

### Backup and Recovery
```bash
# Backup Prometheus data
tar -czf prometheus-backup-$(date +%Y%m%d).tar.gz /opt/prometheus/data

# Backup Grafana dashboards
curl -X GET http://admin:admin@localhost:3000/api/search | jq -r '.[] | .uid' | \
  xargs -I {} curl -X GET http://admin:admin@localhost:3000/api/dashboards/uid/{} > dashboards-backup.json
```

### Updates and Upgrades
- Monitor for security updates to monitoring components
- Test updates in staging before production deployment
- Maintain compatibility between Prometheus, Grafana, and exporters
- Document version dependencies and update procedures

## Contributing

When adding new monitoring components:

1. **Follow naming conventions**: Use consistent metric names and labels
2. **Add documentation**: Update this README and create runbooks
3. **Test thoroughly**: Verify metrics collection and alerting
4. **Consider resource impact**: Monitor resource usage of new components
5. **Update dashboards**: Add relevant visualizations

## Resources

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [ELK Stack Guide](https://www.elastic.co/guide/)
- [Loki Documentation](https://grafana.com/docs/loki/)
- [SRE Best Practices](https://sre.google/books/)
- [Monitoring Best Practices](https://prometheus.io/docs/practices/)

---

For questions or issues with the monitoring setup, please refer to the runbooks or contact the platform engineering team.