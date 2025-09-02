# Disaster Recovery Procedures

## Emergency Response Overview

This document provides step-by-step disaster recovery procedures for the Qdrant MCP workspace. These procedures are designed to restore critical services within 30 minutes (RTO) with maximum 15 minutes of data loss (RPO).

## Incident Classification

### Severity Levels

#### P0 - Critical (Complete Service Outage)
- **Examples**: Total infrastructure failure, data corruption, security breach
- **Response Time**: Immediate (0-15 minutes)
- **Escalation**: All hands, C-level notification
- **Recovery Target**: 30 minutes

#### P1 - High (Partial Service Outage)
- **Examples**: Database failure, major component down, significant performance degradation
- **Response Time**: 15-30 minutes
- **Escalation**: Engineering team, management notification
- **Recovery Target**: 2 hours

#### P2 - Medium (Degraded Service)
- **Examples**: Non-critical service down, minor data inconsistency
- **Response Time**: 30-60 minutes
- **Escalation**: On-call engineer
- **Recovery Target**: 4 hours

#### P3 - Low (Minimal Impact)
- **Examples**: Monitoring alerts, non-critical feature failure
- **Response Time**: Next business day
- **Escalation**: Standard ticket
- **Recovery Target**: 24 hours

## Emergency Contacts and Escalation

### 24/7 On-Call Rotation
```
Primary On-Call:   +1-555-0101 (PagerDuty)
Secondary On-Call: +1-555-0102 (Backup)
Engineering Lead:  +1-555-0103 (Escalation)
Operations Lead:   +1-555-0104 (Infrastructure)
Security Team:     +1-555-0105 (Security incidents)
```

### Communication Channels
- **Primary**: Slack #incident-response
- **Secondary**: Microsoft Teams War Room
- **Status Page**: https://status.company.com
- **Documentation**: Internal Wiki Emergency Procedures

## P0 Critical Recovery Procedures

### Complete Infrastructure Failure

#### Phase 1: Assessment and Immediate Response (0-5 minutes)

1. **Confirm Outage Scope**
```bash
# Check primary monitoring
curl -I https://qdrant-mcp-prod.company.com/health
curl -I https://monitoring.company.com/api/v1/query

# Check secondary monitoring
curl -I https://status.company.com/api/status

# Verify network connectivity
ping -c 3 primary-cluster.company.com
ping -c 3 secondary-cluster.company.com
```

2. **Activate Emergency Response**
```bash
# Trigger incident response
curl -X POST https://pagerduty.api.com/incidents \
  -H "Authorization: Token $PAGERDUTY_TOKEN" \
  -d '{
    "incident": {
      "type": "incident",
      "title": "P0: Complete Qdrant MCP Service Outage",
      "service": {"id": "PXXXXXX", "type": "service"},
      "priority": {"id": "P0XXXXX", "type": "priority"},
      "urgency": "high",
      "body": {"type": "incident_body", "details": "Complete service failure detected"}
    }
  }'

# Update status page
curl -X POST https://api.statuspage.io/v1/pages/$PAGE_ID/incidents \
  -H "Authorization: OAuth $STATUSPAGE_TOKEN" \
  -d '{
    "incident": {
      "name": "Service Outage - Investigating",
      "status": "investigating",
      "impact": "major",
      "body": "We are investigating reports of service unavailability."
    }
  }'
```

3. **Start Communication Protocol**
```bash
# Slack notification
curl -X POST https://hooks.slack.com/services/$SLACK_WEBHOOK \
  -d '{
    "channel": "#incident-response",
    "username": "Emergency Bot",
    "text": "🚨 P0 INCIDENT: Complete Qdrant MCP service outage detected",
    "attachments": [{
      "color": "danger",
      "fields": [
        {"title": "Status", "value": "INVESTIGATING", "short": true},
        {"title": "ETA", "value": "30 minutes", "short": true}
      ]
    }]
  }'
```

#### Phase 2: Failover to Secondary Region (5-15 minutes)

1. **Assess Secondary Region Health**
```bash
# Check secondary Qdrant cluster
./scripts/health-check-secondary.sh

# Verify backup data integrity
./scripts/verification/verify-backup.sh --region=secondary

# Check data lag
./scripts/check-replication-lag.sh
```

2. **Execute Failover**
```bash
#!/bin/bash
# Execute emergency failover to secondary region

echo "🔄 Starting emergency failover to secondary region..."

# Update DNS to point to secondary
aws route53 change-resource-record-sets \
  --hosted-zone-id $HOSTED_ZONE_ID \
  --change-batch '{
    "Changes": [{
      "Action": "UPSERT",
      "ResourceRecordSet": {
        "Name": "qdrant-mcp.company.com",
        "Type": "CNAME",
        "TTL": 60,
        "ResourceRecords": [{"Value": "secondary-qdrant-mcp.company.com"}]
      }
    }]
  }'

# Update load balancer configuration
kubectl patch service qdrant-mcp-service -p '{
  "spec": {
    "selector": {
      "deployment": "secondary-qdrant-mcp"
    }
  }
}'

# Verify failover
sleep 30
curl -f https://qdrant-mcp.company.com/health

if [ $? -eq 0 ]; then
    echo "✅ Failover completed successfully"
    # Update incident status
    ./scripts/update-incident-status.sh "monitoring" "Service restored via failover"
else
    echo "❌ Failover failed, proceeding to backup restoration"
fi
```

#### Phase 3: Full Recovery from Backup (15-30 minutes)

If failover fails, restore from latest backup:

1. **Prepare Recovery Environment**
```bash
#!/bin/bash
# Prepare fresh environment for recovery

echo "🏗️ Preparing recovery environment..."

# Launch new infrastructure
terraform -chdir=./terraform apply \
  -var="environment=recovery" \
  -var="instance_count=3" \
  -auto-approve

# Get recovery cluster endpoints
recovery_endpoints=$(terraform -chdir=./terraform output -json recovery_endpoints)
echo "Recovery cluster ready at: $recovery_endpoints"
```

2. **Restore Qdrant Data**
```bash
#!/bin/bash
# Restore Qdrant from latest backup

RECOVERY_CLUSTER="http://recovery-qdrant:6333"
LATEST_BACKUP=$(ls -t /backups/qdrant/full | head -1)

echo "🔄 Restoring Qdrant data from backup: $LATEST_BACKUP"

# Extract backup metadata
cd "/backups/qdrant/full/$LATEST_BACKUP"

# Restore each collection
for snapshot_file in *.snapshot; do
    collection_name=$(basename "$snapshot_file" .snapshot)
    meta_file="${collection_name}.meta"
    
    echo "Restoring collection: $collection_name"
    
    # Get collection configuration from metadata
    collection_config=$(cat "$meta_file" | jq '.collection_config')
    
    # Create collection
    curl -X PUT "$RECOVERY_CLUSTER/collections/$collection_name" \
      -H "Content-Type: application/json" \
      -d "$collection_config"
    
    # Upload and restore snapshot
    curl -X POST "$RECOVERY_CLUSTER/collections/$collection_name/snapshots/upload" \
      -F "snapshot=@$snapshot_file"
    
    # Get uploaded snapshot name
    snapshot_name=$(curl -s "$RECOVERY_CLUSTER/collections/$collection_name/snapshots" | \
      jq -r '.result[-1].name')
    
    # Restore from snapshot
    curl -X PUT "$RECOVERY_CLUSTER/collections/$collection_name/snapshots/$snapshot_name/recover"
    
    echo "✅ Collection $collection_name restored"
done

echo "🎉 Qdrant data restoration completed"
```

3. **Restore Application Configuration**
```bash
#!/bin/bash
# Restore application configuration

LATEST_CONFIG_BACKUP=$(ls -t /backups/config | head -1)
CONFIG_BACKUP_DIR="/backups/config/$LATEST_CONFIG_BACKUP"

echo "🔧 Restoring application configuration..."

# Restore Kubernetes configurations
kubectl apply -f "$CONFIG_BACKUP_DIR/configmaps.yaml"
kubectl apply -f "$CONFIG_BACKUP_DIR/secrets.yaml"

# Update application deployment to use recovery cluster
kubectl patch deployment qdrant-mcp-app -p '{
  "spec": {
    "template": {
      "spec": {
        "containers": [{
          "name": "qdrant-mcp",
          "env": [{
            "name": "QDRANT_URL",
            "value": "http://recovery-qdrant:6333"
          }]
        }]
      }
    }
  }
}'

# Wait for deployment rollout
kubectl rollout status deployment/qdrant-mcp-app --timeout=300s

echo "✅ Configuration restoration completed"
```

4. **Final Verification and Cutover**
```bash
#!/bin/bash
# Final verification before production cutover

echo "🧪 Running recovery verification tests..."

# Test Qdrant connectivity
./scripts/testing/test-qdrant-connectivity.sh --endpoint="$RECOVERY_CLUSTER"

# Test application functionality
./scripts/testing/test-application-endpoints.sh --base-url="http://recovery-app"

# Run smoke tests
./scripts/testing/smoke-tests.sh --environment=recovery

# Verify data integrity
./scripts/verification/full-verification.sh --environment=recovery

if [ $? -eq 0 ]; then
    echo "✅ All verification tests passed"
    
    # Cutover to recovery environment
    ./scripts/cutover-to-recovery.sh
    
    # Update incident status
    ./scripts/update-incident-status.sh "resolved" "Service fully restored from backup"
    
    echo "🎉 Disaster recovery completed successfully"
else
    echo "❌ Verification failed - manual intervention required"
    exit 1
fi
```

## Partial Service Recovery (P1)

### Qdrant Database Failure Only

1. **Isolate Failed Components**
```bash
# Stop failing Qdrant instance
kubectl scale deployment qdrant-primary --replicas=0

# Route traffic to healthy instances
kubectl patch service qdrant-service -p '{
  "spec": {
    "selector": {
      "app": "qdrant-backup"
    }
  }
}'
```

2. **Restore Qdrant from Latest Snapshot**
```bash
# Get latest incremental backup
LATEST_INCREMENTAL=$(ls -t /backups/qdrant/incremental | head -1)

# Restore specific collections that failed
./scripts/recovery/restore-collections.sh \
  --backup-dir="/backups/qdrant/incremental/$LATEST_INCREMENTAL" \
  --collections="user-docs,embeddings"
```

3. **Verify and Reconnect**
```bash
# Test restored Qdrant instance
./scripts/testing/test-qdrant-collections.sh

# Gradually route traffic back
kubectl patch service qdrant-service -p '{
  "spec": {
    "selector": {
      "app": "qdrant-primary"
    }
  }
}'

# Scale back up
kubectl scale deployment qdrant-primary --replicas=3
```

## Data Corruption Recovery

### Detecting Data Corruption

1. **Automated Corruption Detection**
```bash
#!/bin/bash
# Run data integrity checks

# Check Qdrant collection health
collections=$(curl -s "http://localhost:6333/collections" | jq -r '.result.collections[].name')

for collection in $collections; do
    # Get collection info
    info=$(curl -s "http://localhost:6333/collections/$collection")
    status=$(echo "$info" | jq -r '.result.status')
    
    if [ "$status" != "green" ]; then
        echo "⚠️ Collection $collection status: $status"
        
        # Try to recover from corruption
        ./scripts/recovery/repair-collection.sh "$collection"
    fi
done

# Check for data consistency
./scripts/verification/consistency-check.sh
```

2. **Point-in-Time Recovery**
```bash
#!/bin/bash
# Restore to specific backup timestamp

TARGET_TIMESTAMP="2024-01-15T14:30:00Z"
echo "🕰️ Restoring to timestamp: $TARGET_TIMESTAMP"

# Find backup closest to target time
find_closest_backup() {
    local target_epoch=$(date -d "$TARGET_TIMESTAMP" +%s)
    local closest_backup=""
    local min_diff=999999999
    
    for backup_dir in /backups/qdrant/full/*; do
        backup_date=$(basename "$backup_dir" | cut -d'_' -f1-2)
        backup_epoch=$(date -d "${backup_date:0:8} ${backup_date:9:2}:${backup_date:11:2}:00" +%s 2>/dev/null || echo 0)
        
        diff=$((target_epoch - backup_epoch))
        if [ $diff -ge 0 ] && [ $diff -lt $min_diff ]; then
            min_diff=$diff
            closest_backup="$backup_dir"
        fi
    done
    
    echo "$closest_backup"
}

RECOVERY_BACKUP=$(find_closest_backup)
echo "Using backup: $RECOVERY_BACKUP"

# Restore from selected backup
./scripts/recovery/restore-from-backup.sh --backup-dir="$RECOVERY_BACKUP" --verify-integrity
```

## Network and Infrastructure Recovery

### Network Isolation/Partition

1. **Detect Network Issues**
```bash
# Check inter-service connectivity
./scripts/network/connectivity-test.sh

# Check external dependencies
./scripts/network/external-deps-test.sh

# Verify DNS resolution
./scripts/network/dns-test.sh
```

2. **Network Recovery Actions**
```bash
# Restart network services
systemctl restart networking
systemctl restart docker

# Flush DNS cache
systemctl flush-dns

# Reset iptables rules
./scripts/network/reset-firewall-rules.sh

# Verify connectivity restoration
./scripts/network/full-connectivity-test.sh
```

## Security Incident Recovery

### Data Breach Response

1. **Immediate Containment**
```bash
# Isolate affected systems
./scripts/security/isolate-systems.sh --severity=high

# Disable compromised accounts
./scripts/security/disable-accounts.sh --from-file=compromised-accounts.txt

# Enable enhanced monitoring
./scripts/security/enable-forensic-mode.sh
```

2. **Forensic Backup Creation**
```bash
# Create forensic backup before any changes
./scripts/security/forensic-backup.sh --incident-id="INC-2024-001"

# Document system state
./scripts/security/capture-system-state.sh
```

3. **Clean Recovery**
```bash
# Restore from clean backup (pre-incident)
CLEAN_BACKUP=$(./scripts/security/find-clean-backup.sh --before="$INCIDENT_TIME")
./scripts/recovery/restore-from-backup.sh --backup-dir="$CLEAN_BACKUP" --security-mode

# Update all credentials
./scripts/security/rotate-all-credentials.sh

# Apply security patches
./scripts/security/apply-emergency-patches.sh
```

## Testing and Validation

### Recovery Verification Checklist

```bash
#!/bin/bash
# Comprehensive recovery verification

echo "🧪 Starting recovery verification..."

# Test critical paths
./scripts/testing/critical-path-tests.sh

# Verify data integrity
./scripts/verification/data-integrity-check.sh

# Performance baseline test
./scripts/testing/performance-baseline.sh

# Security posture check
./scripts/security/security-posture-check.sh

# User acceptance test
./scripts/testing/user-acceptance-test.sh

# Generate recovery report
./scripts/reporting/generate-recovery-report.sh
```

### Post-Recovery Actions

1. **Monitor for Stability**
```bash
# Enhanced monitoring for 24 hours
./scripts/monitoring/enable-enhanced-monitoring.sh --duration=24h

# Watch for anomalies
./scripts/monitoring/anomaly-detection.sh --sensitivity=high
```

2. **Update Stakeholders**
```bash
# Send recovery completion notification
./scripts/communication/send-recovery-notification.sh

# Update status page
./scripts/communication/update-status-page.sh --status=resolved

# Schedule post-incident review
./scripts/communication/schedule-post-mortem.sh --incident-id="$INCIDENT_ID"
```

## Recovery Time Objectives (RTO)

| Recovery Scenario | Target RTO | Maximum RTO | Automated? |
|------------------|------------|-------------|-----------|
| Qdrant node failure | 5 minutes | 15 minutes | Yes |
| Application pod failure | 2 minutes | 5 minutes | Yes |
| Full cluster failure | 15 minutes | 30 minutes | Partial |
| Data corruption | 30 minutes | 2 hours | No |
| Security breach | 1 hour | 4 hours | No |
| Complete DC failure | 30 minutes | 1 hour | Yes |

## Recovery Point Objectives (RPO)

| Data Type | Target RPO | Maximum RPO | Backup Frequency |
|-----------|------------|-------------|------------------|
| Qdrant collections | 5 minutes | 15 minutes | Continuous replication |
| User data | 15 minutes | 1 hour | Every 15 minutes |
| Configuration | 1 hour | 4 hours | Every hour |
| Logs | 24 hours | 48 hours | Daily |

## Communication Templates

### Initial Incident Notification
```
🚨 INCIDENT ALERT 🚨

Incident ID: INC-YYYY-MM-DD-XXX
Severity: P0 - Critical
Service: Qdrant MCP Workspace
Status: INVESTIGATING

Issue: [Brief description]
Impact: [User impact description]
ETA: [Estimated resolution time]

Updates will be provided every 15 minutes.
Status page: https://status.company.com
```

### Resolution Notification
```
✅ INCIDENT RESOLVED ✅

Incident ID: INC-YYYY-MM-DD-XXX
Resolution Time: [Duration]
Root Cause: [Brief cause]

Services have been restored to full functionality.
A post-incident review will be scheduled within 48 hours.

Thank you for your patience.
```

---

These disaster recovery procedures provide comprehensive guidance for handling various failure scenarios while maintaining the 30-minute RTO and 15-minute RPO targets. Regular testing and updates ensure procedures remain effective.