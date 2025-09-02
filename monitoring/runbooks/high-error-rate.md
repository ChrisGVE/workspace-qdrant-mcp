# Runbook: High Error Rate

## Alert Information
- **Alert Name**: HighErrorRate
- **Severity**: Critical
- **Threshold**: Error rate > 5% for 2+ minutes
- **SLA Impact**: Yes - affects service availability

## Description
This alert fires when the HTTP error rate (5xx responses) exceeds 5% of total requests for more than 2 minutes. This typically indicates a serious application or infrastructure issue that requires immediate attention.

## Immediate Response (First 5 minutes)

### 1. Assess Scope and Impact
```bash
# Check overall service health
curl -s http://localhost:8080/health | jq '.'

# Check current error rate in Grafana or via Prometheus
# Query: rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m]) * 100

# Check if specific endpoints are affected
# Query: rate(http_requests_total{status=~"5.."}[5m]) by (endpoint)
```

### 2. Check Recent Deployments
```bash
# Check recent git commits
git log --oneline -10

# Check deployment logs
kubectl logs deployment/workspace-qdrant-mcp --tail=50

# Check container restarts
kubectl get pods -l app=workspace-qdrant-mcp
```

### 3. Verify Dependencies
```bash
# Check Qdrant database connectivity
curl -s http://localhost:6333/cluster

# Check database connection pool status
# Look for connection timeout errors in logs
grep -i "connection" /var/log/workspace-qdrant-mcp/*.log | tail -20
```

## Investigation Steps

### 4. Analyze Error Patterns
```bash
# Check error logs for patterns
tail -100 /var/log/workspace-qdrant-mcp/error.log | grep -E "(ERROR|CRITICAL)"

# Look for specific error types
grep -E "(TimeoutError|ConnectionError|MemoryError)" /var/log/workspace-qdrant-mcp/*.log | tail -20

# Check Grafana logs dashboard for error categorization
# URL: http://localhost:3000/d/logs-dashboard
```

### 5. Resource Utilization Check
```bash
# Check system resources
htop
df -h
free -h

# Check application memory usage
ps aux | grep workspace-qdrant-mcp | head -5

# Check for memory leaks
# Query in Prometheus: process_resident_memory_bytes
```

### 6. Performance Metrics Analysis
```bash
# Check response times
# Prometheus query: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))

# Check queue backlog
# Prometheus query: task_queue_size

# Check database query performance
# Look for slow query logs
```

## Common Causes and Solutions

### Database Connection Issues
**Symptoms**: Connection timeout errors, database connection pool exhausted
```bash
# Check database status
curl -s http://localhost:6333/cluster | jq '.status'

# Restart database connection pool (if using connection pooling)
# Application-specific restart command

# Check database resource usage
# Monitor Qdrant metrics at http://localhost:6333/metrics
```

### Memory Issues
**Symptoms**: OutOfMemoryError, high memory usage alerts
```bash
# Check memory usage breakdown
ps aux --sort=-%mem | head -10

# Check for memory leaks in application
# Restart application if necessary
kubectl rollout restart deployment/workspace-qdrant-mcp

# Increase memory limits (temporary fix)
kubectl patch deployment workspace-qdrant-mcp -p '{"spec":{"template":{"spec":{"containers":[{"name":"app","resources":{"limits":{"memory":"2Gi"}}}]}}}}'
```

### Code Issues
**Symptoms**: Specific error patterns, recent deployment correlation
```bash
# Rollback to previous version
kubectl rollout undo deployment/workspace-qdrant-mcp

# Check application logs for stack traces
grep -A 20 "Traceback" /var/log/workspace-qdrant-mcp/*.log | tail -50

# Review recent code changes
git diff HEAD~3 HEAD
```

### External Service Dependencies
**Symptoms**: Timeout errors, external API failures
```bash
# Check external service health
curl -I https://external-api.com/health

# Review circuit breaker status (if implemented)
# Check retry logic and timeout configurations
```

## Resolution Actions

### Immediate Mitigation
1. **Scale up resources** (if resource-related):
   ```bash
   kubectl scale deployment workspace-qdrant-mcp --replicas=5
   ```

2. **Rollback deployment** (if deployment-related):
   ```bash
   kubectl rollout undo deployment/workspace-qdrant-mcp
   kubectl rollout status deployment/workspace-qdrant-mcp
   ```

3. **Restart services** (if connection-related):
   ```bash
   kubectl rollout restart deployment/workspace-qdrant-mcp
   kubectl rollout restart deployment/qdrant
   ```

### Long-term Fixes
1. **Fix identified code issues** and deploy with proper testing
2. **Increase resource limits** based on usage patterns
3. **Implement circuit breakers** for external dependencies
4. **Add retry logic** with exponential backoff
5. **Improve error handling** and logging

## Verification Steps
After implementing fixes:

```bash
# Monitor error rate for 10+ minutes
# Query: rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m]) * 100

# Check that error rate is below 1%
# Verify response times are normal
# Query: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))

# Confirm no new errors in logs
tail -f /var/log/workspace-qdrant-mcp/*.log

# Run health checks
curl -s http://localhost:8080/health | jq '.status'
```

## Communication Template

### For Internal Team
```
🚨 HIGH ERROR RATE INCIDENT

Status: INVESTIGATING/MITIGATING/RESOLVED
Start Time: [TIME]
Scope: Error rate at [X%] affecting [Y] endpoints
Impact: [Customer impact description]
Current Action: [What's being done]
ETA: [Expected resolution time]
```

### For Customers (if external-facing)
```
We're currently experiencing elevated error rates that may affect service performance. 
Our team is actively investigating and working on a resolution. 
We'll provide updates every 15 minutes until resolved.
```

## Post-Incident Tasks

### Immediate (within 2 hours)
- [ ] Verify error rate remains < 1%
- [ ] Confirm all systems are stable
- [ ] Review and document root cause
- [ ] Update monitoring thresholds if needed

### Follow-up (within 24 hours)
- [ ] Conduct post-mortem meeting
- [ ] Create action items for prevention
- [ ] Update runbook with lessons learned
- [ ] Implement additional monitoring if needed

### Long-term (within 1 week)
- [ ] Implement preventive measures
- [ ] Update SLA/SLO definitions
- [ ] Review and test incident response procedures
- [ ] Share learnings with broader team

## Related Runbooks
- [Service Down](./service-down.md)
- [High Memory Usage](./high-memory.md)
- [Database Connection Failures](./db-connection-failures.md)
- [Slow Response Times](./slow-response.md)

## Emergency Contacts
- **Primary On-Call**: [Contact Info]
- **Secondary On-Call**: [Contact Info]
- **Engineering Manager**: [Contact Info]
- **Database Team**: [Contact Info]

## Additional Resources
- [Grafana Dashboard](http://localhost:3000/d/application-overview)
- [Prometheus Alerts](http://localhost:9093)
- [Service Documentation](../docs/service-architecture.md)
- [Deployment Procedures](../docs/deployment.md)