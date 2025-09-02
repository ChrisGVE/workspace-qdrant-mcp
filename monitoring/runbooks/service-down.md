# Runbook: Service Down

## Alert Information
- **Alert Name**: ServiceDown
- **Severity**: Critical
- **Threshold**: Service unreachable for 30+ seconds
- **SLA Impact**: Yes - complete service outage

## Description
This alert fires when the service is completely unreachable or returning consistent failures on health checks. This represents a complete outage requiring immediate response.

## Immediate Response (First 2 minutes)

### 1. Confirm Outage Scope
```bash
# Test service connectivity from multiple locations
curl -m 5 http://localhost:8000/health
curl -m 5 http://localhost:8000/ready
curl -m 5 http://localhost:8080/alive

# Check if this is a monitoring false positive
# Test from external monitoring service or different network
```

### 2. Check Service Status
```bash
# Check container/process status
kubectl get pods -l app=workspace-qdrant-mcp
docker ps | grep workspace-qdrant-mcp

# Check service logs for crashes
kubectl logs deployment/workspace-qdrant-mcp --tail=50
docker logs workspace-qdrant-mcp --tail=50

# Check system resource availability
df -h
free -h
uptime
```

### 3. Escalate Immediately
- Page primary on-call engineer
- Notify incident response team
- Update status page (if external service)

## Investigation Steps (First 5 minutes)

### 4. Analyze Recent Changes
```bash
# Check recent deployments
kubectl describe deployment workspace-qdrant-mcp
kubectl rollout history deployment/workspace-qdrant-mcp

# Check recent configuration changes
git log --oneline --since="2 hours ago"

# Check infrastructure changes
# Review cloud provider console for recent changes
```

### 5. Examine System Resources
```bash
# Check critical system resources
top
iostat 1 5
netstat -tulpn | grep :8000

# Check disk space (common cause of service failures)
df -h
du -sh /var/log/* | sort -hr | head -10

# Check system logs
journalctl -u workspace-qdrant-mcp --since "1 hour ago" --no-pager
tail -50 /var/log/syslog
```

### 6. Check Dependencies
```bash
# Verify database connectivity
curl -s http://localhost:6333/cluster
telnet localhost 6333

# Check external service dependencies
ping google.com
nslookup external-dependency.com

# Check network connectivity
netstat -i
ip route show
```

## Common Causes and Immediate Fixes

### Application Crash/OOM Kill
**Symptoms**: Container restarts, OOM messages in logs, high memory usage before crash
```bash
# Check for OOM kills
dmesg | grep -i "killed process\|out of memory"
journalctl -k | grep -i "out of memory"

# Check container restart count
kubectl get pods -l app=workspace-qdrant-mcp -o wide

# Immediate fix: Restart with higher memory limits
kubectl patch deployment workspace-qdrant-mcp -p '{"spec":{"template":{"spec":{"containers":[{"name":"app","resources":{"limits":{"memory":"4Gi"}}}]}}}}'
kubectl rollout status deployment/workspace-qdrant-mcp
```

### Port/Binding Issues
**Symptoms**: Port already in use, binding errors
```bash
# Check what's using the port
lsof -i :8000
netstat -tulpn | grep :8000

# Kill conflicting process if safe
sudo kill $(lsof -ti:8000)

# Restart service
kubectl rollout restart deployment/workspace-qdrant-mcp
```

### Database Connectivity Loss
**Symptoms**: Database connection errors in logs
```bash
# Check Qdrant status
curl -s http://localhost:6333/cluster | jq '.'
kubectl get pods -l app=qdrant

# Restart database if needed
kubectl rollout restart deployment/qdrant
kubectl wait --for=condition=ready pod -l app=qdrant --timeout=120s

# Restart application after database is ready
kubectl rollout restart deployment/workspace-qdrant-mcp
```

### Configuration Issues
**Symptoms**: Configuration validation errors, missing environment variables
```bash
# Check environment variables
kubectl describe deployment workspace-qdrant-mcp | grep -A 20 "Environment:"

# Check configmap/secrets
kubectl get configmap workspace-qdrant-mcp-config -o yaml
kubectl get secret workspace-qdrant-mcp-secrets -o yaml

# Validate configuration files
# Check for syntax errors in config files
```

### Resource Exhaustion
**Symptoms**: High CPU/memory usage, slow response before failure
```bash
# Check node resources
kubectl describe node
kubectl top nodes

# Scale up resources immediately
kubectl scale deployment workspace-qdrant-mcp --replicas=3

# Check for resource quotas
kubectl describe quota --all-namespaces
```

## Recovery Actions

### Option 1: Quick Restart (if simple issue)
```bash
# Restart application
kubectl rollout restart deployment/workspace-qdrant-mcp
kubectl rollout status deployment/workspace-qdrant-mcp --timeout=300s

# Verify service is responding
curl -s http://localhost:8000/health | jq '.status'
```

### Option 2: Rollback (if recent deployment issue)
```bash
# Rollback to previous version
kubectl rollout undo deployment/workspace-qdrant-mcp
kubectl rollout status deployment/workspace-qdrant-mcp --timeout=300s

# Verify rollback success
curl -s http://localhost:8000/health
kubectl get deployment workspace-qdrant-mcp -o wide
```

### Option 3: Emergency Scaling (if resource issue)
```bash
# Scale out horizontally
kubectl scale deployment workspace-qdrant-mcp --replicas=5

# Scale up vertically
kubectl patch deployment workspace-qdrant-mcp -p '{
  "spec": {
    "template": {
      "spec": {
        "containers": [{
          "name": "app",
          "resources": {
            "limits": {"cpu": "2", "memory": "4Gi"},
            "requests": {"cpu": "1", "memory": "2Gi"}
          }
        }]
      }
    }
  }
}'
```

### Option 4: Full Infrastructure Recovery
```bash
# If entire cluster/infrastructure is affected
# Bring up backup infrastructure
# Restore from latest backup
# Update DNS to point to backup environment
```

## Verification and Monitoring

### Immediate Verification (within 2 minutes of fix)
```bash
# Test basic functionality
curl -s http://localhost:8000/health
curl -s http://localhost:8000/ready

# Run basic smoke tests
# Test critical user journeys
curl -X POST http://localhost:8000/api/search -d '{"query": "test"}'

# Check error rates
# Monitor Grafana for error rate returning to normal
```

### Extended Monitoring (next 15 minutes)
```bash
# Monitor service stability
watch -n 10 'curl -s http://localhost:8000/health | jq ".status"'

# Check resource usage
kubectl top pods -l app=workspace-qdrant-mcp

# Monitor logs for any issues
kubectl logs -f deployment/workspace-qdrant-mcp | grep -E "(ERROR|CRITICAL|WARN)"
```

## Communication Templates

### Initial Alert (within 2 minutes)
```
🚨 CRITICAL OUTAGE - Service Down

Service: Workspace Qdrant MCP
Status: Complete Outage
Start Time: [TIMESTAMP]
Impact: All users unable to access service
Response: Engineering team responding
ETA: Investigating - updates in 5 minutes
```

### Update Template (every 5 minutes until resolved)
```
📊 OUTAGE UPDATE

Status: [INVESTIGATING/IDENTIFIED/FIXING/MONITORING]
Cause: [Root cause if identified]
Current Action: [What's being done now]
Progress: [% complete if applicable]
Next Update: [Time]
```

### Resolution Notification
```
✅ OUTAGE RESOLVED

Service: Workspace Qdrant MCP
Duration: [XX minutes]
Root Cause: [Brief description]
Fix Applied: [What was done]
Status: Fully operational
Monitoring: Continuing to monitor for stability
```

## Post-Incident Actions

### Immediate (within 1 hour)
- [ ] Verify service is fully stable
- [ ] Document timeline and actions taken
- [ ] Check if any data was lost or corrupted
- [ ] Review and update monitoring if needed

### Short-term (within 24 hours)
- [ ] Conduct post-mortem meeting
- [ ] Create detailed incident report
- [ ] Identify and prioritize prevention measures
- [ ] Update runbooks based on learnings
- [ ] Test backup and recovery procedures

### Long-term (within 1 week)
- [ ] Implement preventive measures
- [ ] Update SLA/SLO targets if needed
- [ ] Improve monitoring and alerting
- [ ] Conduct chaos engineering tests
- [ ] Update disaster recovery procedures

## Prevention Strategies

### Monitoring Improvements
- Implement additional health checks
- Add synthetic monitoring from multiple locations
- Set up predictive alerts for resource exhaustion
- Monitor dependency health more closely

### Infrastructure Resilience
- Implement auto-scaling policies
- Set up multi-region failover
- Improve backup and recovery procedures
- Implement circuit breakers for dependencies

### Operational Practices
- Implement blue-green deployments
- Improve testing in staging environments
- Regular chaos engineering exercises
- Better change management procedures

## Emergency Contacts
- **Primary On-Call**: [Contact Info]
- **Backup On-Call**: [Contact Info]
- **Engineering Manager**: [Contact Info]
- **Infrastructure Team**: [Contact Info]
- **Executive Escalation**: [Contact Info]

## Related Runbooks
- [High Error Rate](./high-error-rate.md)
- [Database Connection Failures](./db-connection-failures.md)
- [High Memory Usage](./high-memory.md)
- [Infrastructure Failure](./infrastructure-failure.md)

## Quick Reference Commands
```bash
# Service status
kubectl get pods -l app=workspace-qdrant-mcp
curl -s http://localhost:8000/health

# Restart service
kubectl rollout restart deployment/workspace-qdrant-mcp

# Rollback deployment
kubectl rollout undo deployment/workspace-qdrant-mcp

# Scale service
kubectl scale deployment workspace-qdrant-mcp --replicas=3

# View logs
kubectl logs -f deployment/workspace-qdrant-mcp

# Check resources
kubectl top pods -l app=workspace-qdrant-mcp
```