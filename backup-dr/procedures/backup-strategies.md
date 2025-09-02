# Backup Strategies

## Overview

This document outlines comprehensive backup strategies for the Qdrant MCP workspace, covering all critical components including vector databases, configurations, user data, and infrastructure components.

## Backup Architecture

### Data Classification

#### Critical Data (RPO: 15 minutes)
- **Qdrant Collections**: Vector embeddings and metadata
- **User Sessions**: Active user state and preferences  
- **Transaction Logs**: Operation history and audit trails
- **Security Keys**: Authentication and encryption keys

#### Important Data (RPO: 1 hour)
- **Configuration Files**: Application and system configurations
- **User Data**: Documents, uploads, and generated content
- **Analytics Data**: Usage metrics and performance data
- **Cache Data**: Frequently accessed computed results

#### Supplementary Data (RPO: 24 hours)
- **Application Logs**: Historical log data
- **Temporary Files**: Processing intermediates
- **Development Data**: Non-production datasets
- **Archive Data**: Historical backups and reports

## Qdrant Vector Database Backup

### Snapshot-based Backup

Qdrant provides native snapshot functionality for consistent backups:

```bash
# Create collection snapshot
curl -X POST "http://localhost:6333/collections/{collection_name}/snapshots"

# Download snapshot
curl "http://localhost:6333/collections/{collection_name}/snapshots/{snapshot_name}" \
  --output "/backups/qdrant/{collection_name}/{snapshot_name}.snapshot"

# List available snapshots
curl "http://localhost:6333/collections/{collection_name}/snapshots"
```

### Full Database Backup Strategy

#### Daily Full Backup (2:00 AM UTC)
```bash
#!/bin/bash
# Full Qdrant backup with all collections

BACKUP_DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/qdrant/full/${BACKUP_DATE}"

# Create backup directory
mkdir -p "${BACKUP_DIR}"

# Get all collections
collections=$(curl -s "http://localhost:6333/collections" | jq -r '.result.collections[].name')

for collection in $collections; do
    echo "Backing up collection: $collection"
    
    # Create snapshot
    snapshot_response=$(curl -s -X POST "http://localhost:6333/collections/${collection}/snapshots")
    snapshot_name=$(echo "$snapshot_response" | jq -r '.result.name')
    
    # Download snapshot
    curl -s "http://localhost:6333/collections/${collection}/snapshots/${snapshot_name}" \
        --output "${BACKUP_DIR}/${collection}.snapshot"
    
    # Verify snapshot integrity
    if [ $? -eq 0 ]; then
        echo "✓ Collection $collection backed up successfully"
        
        # Generate metadata
        cat > "${BACKUP_DIR}/${collection}.meta" << EOF
{
  "collection_name": "${collection}",
  "snapshot_name": "${snapshot_name}",
  "backup_date": "${BACKUP_DATE}",
  "file_size": $(stat -f%z "${BACKUP_DIR}/${collection}.snapshot"),
  "checksum": "$(sha256sum "${BACKUP_DIR}/${collection}.snapshot" | cut -d' ' -f1)"
}
EOF
    else
        echo "✗ Failed to backup collection $collection"
        exit 1
    fi
done

# Create backup manifest
cat > "${BACKUP_DIR}/manifest.json" << EOF
{
  "backup_type": "full",
  "backup_date": "${BACKUP_DATE}",
  "collections_count": $(echo "$collections" | wc -l),
  "backup_size": $(du -sb "${BACKUP_DIR}" | cut -f1),
  "created_by": "automated-backup",
  "qdrant_version": "$(curl -s http://localhost:6333/ | jq -r '.version')"
}
EOF

echo "Full backup completed: ${BACKUP_DIR}"
```

#### Incremental Backup (Every 4 hours)
```bash
#!/bin/bash
# Incremental Qdrant backup - only changed collections

LAST_BACKUP=$(find /backups/qdrant/incremental -name "*.snapshot" -newer /var/lib/backup/last-full-backup.timestamp)
BACKUP_DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/qdrant/incremental/${BACKUP_DATE}"

mkdir -p "${BACKUP_DIR}"

# Check which collections have been modified
collections=$(curl -s "http://localhost:6333/collections" | jq -r '.result.collections[].name')

for collection in $collections; do
    # Check collection info for recent modifications
    collection_info=$(curl -s "http://localhost:6333/collections/${collection}")
    points_count=$(echo "$collection_info" | jq -r '.result.points_count')
    
    # Check if collection changed since last backup
    last_backup_points="/var/lib/backup/collection-states/${collection}.count"
    
    if [ ! -f "$last_backup_points" ] || [ "$points_count" != "$(cat $last_backup_points)" ]; then
        echo "Changes detected in collection: $collection"
        
        # Perform incremental backup
        snapshot_response=$(curl -s -X POST "http://localhost:6333/collections/${collection}/snapshots")
        snapshot_name=$(echo "$snapshot_response" | jq -r '.result.name')
        
        curl -s "http://localhost:6333/collections/${collection}/snapshots/${snapshot_name}" \
            --output "${BACKUP_DIR}/${collection}.snapshot"
        
        # Update state tracking
        echo "$points_count" > "$last_backup_points"
        
        echo "✓ Incremental backup completed for $collection"
    else
        echo "→ No changes in collection $collection, skipping"
    fi
done
```

### Cross-Region Replication

For disaster recovery, implement real-time replication to secondary regions:

```bash
#!/bin/bash
# Cross-region Qdrant replication setup

PRIMARY_QDRANT="http://primary-qdrant:6333"
SECONDARY_QDRANT="http://secondary-qdrant:6333"

# Replicate collection to secondary region
replicate_collection() {
    local collection_name=$1
    
    # Get collection configuration from primary
    collection_config=$(curl -s "${PRIMARY_QDRANT}/collections/${collection_name}")
    
    # Create collection on secondary (if not exists)
    curl -X PUT "${SECONDARY_QDRANT}/collections/${collection_name}" \
        -H "Content-Type: application/json" \
        -d "$collection_config"
    
    # Create and transfer snapshot
    snapshot_response=$(curl -s -X POST "${PRIMARY_QDRANT}/collections/${collection_name}/snapshots")
    snapshot_name=$(echo "$snapshot_response" | jq -r '.result.name')
    
    # Download snapshot from primary
    curl -s "${PRIMARY_QDRANT}/collections/${collection_name}/snapshots/${snapshot_name}" \
        --output "/tmp/${collection_name}.snapshot"
    
    # Upload snapshot to secondary
    curl -X POST "${SECONDARY_QDRANT}/collections/${collection_name}/snapshots/upload" \
        -F "snapshot=@/tmp/${collection_name}.snapshot"
    
    # Restore snapshot on secondary
    curl -X PUT "${SECONDARY_QDRANT}/collections/${collection_name}/snapshots/${snapshot_name}/recover"
    
    echo "✓ Collection $collection_name replicated to secondary region"
}

# Replicate all collections
collections=$(curl -s "${PRIMARY_QDRANT}/collections" | jq -r '.result.collections[].name')
for collection in $collections; do
    replicate_collection "$collection"
done
```

## Configuration Backup

### Application Configuration
```bash
#!/bin/bash
# Backup application configuration

CONFIG_BACKUP_DIR="/backups/config/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$CONFIG_BACKUP_DIR"

# Backup environment files
cp -r .env* "$CONFIG_BACKUP_DIR/"

# Backup Docker configurations
cp docker-compose*.yml "$CONFIG_BACKUP_DIR/"

# Backup Kubernetes manifests
kubectl get configmaps -o yaml > "$CONFIG_BACKUP_DIR/configmaps.yaml"
kubectl get secrets -o yaml > "$CONFIG_BACKUP_DIR/secrets.yaml"

# Backup MCP configuration
cp .mcp.json "$CONFIG_BACKUP_DIR/"

# Backup monitoring configuration
cp -r monitoring/ "$CONFIG_BACKUP_DIR/"

# Create configuration manifest
cat > "$CONFIG_BACKUP_DIR/manifest.json" << EOF
{
  "backup_type": "configuration",
  "backup_date": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "components": [
    "environment_files",
    "docker_compose",
    "kubernetes_manifests",
    "mcp_configuration", 
    "monitoring_configuration"
  ]
}
EOF

echo "Configuration backup completed: $CONFIG_BACKUP_DIR"
```

### Infrastructure as Code Backup
```bash
#!/bin/bash
# Backup infrastructure definitions

IaC_BACKUP_DIR="/backups/infrastructure/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$IaC_BACKUP_DIR"

# Backup Terraform configurations
if [ -d "terraform/" ]; then
    cp -r terraform/ "$IaC_BACKUP_DIR/"
    
    # Export Terraform state
    cd terraform && terraform state pull > "$IaC_BACKUP_DIR/terraform.tfstate"
    cd ..
fi

# Backup Ansible playbooks
if [ -d "ansible/" ]; then
    cp -r ansible/ "$IaC_BACKUP_DIR/"
fi

# Backup Helm charts
if [ -d "helm/" ]; then
    cp -r helm/ "$IaC_BACKUP_DIR/"
fi

# Backup container images list
docker images --format "table {{.Repository}}:{{.Tag}}" > "$IaC_BACKUP_DIR/docker-images.list"

# Backup Kubernetes cluster state
kubectl cluster-info dump --output-directory="$IaC_BACKUP_DIR/cluster-dump"

echo "Infrastructure backup completed: $IaC_BACKUP_DIR"
```

## User Data Backup

### Application State Backup
```bash
#!/bin/bash
# Backup user data and application state

USER_DATA_DIR="/backups/userdata/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$USER_DATA_DIR"

# Backup user uploads
if [ -d "/app/uploads" ]; then
    rsync -av /app/uploads/ "$USER_DATA_DIR/uploads/"
fi

# Backup user sessions (if using file-based sessions)
if [ -d "/app/sessions" ]; then
    cp -r /app/sessions/ "$USER_DATA_DIR/"
fi

# Backup application cache
if [ -d "/app/cache" ]; then
    cp -r /app/cache/ "$USER_DATA_DIR/"
fi

# Backup application logs
journalctl --unit=qdrant-mcp --since="24 hours ago" > "$USER_DATA_DIR/application.log"

echo "User data backup completed: $USER_DATA_DIR"
```

## Backup Verification and Testing

### Integrity Verification
```bash
#!/bin/bash
# Verify backup integrity

verify_backup() {
    local backup_dir=$1
    local backup_type=$2
    
    echo "Verifying $backup_type backup: $backup_dir"
    
    # Check manifest exists
    if [ ! -f "$backup_dir/manifest.json" ]; then
        echo "✗ Missing manifest file"
        return 1
    fi
    
    # Verify file checksums
    case $backup_type in
        "qdrant")
            for meta_file in "$backup_dir"/*.meta; do
                if [ -f "$meta_file" ]; then
                    expected_checksum=$(jq -r '.checksum' "$meta_file")
                    collection_file="${meta_file%.meta}.snapshot"
                    actual_checksum=$(sha256sum "$collection_file" | cut -d' ' -f1)
                    
                    if [ "$expected_checksum" = "$actual_checksum" ]; then
                        echo "✓ $(basename $collection_file) checksum verified"
                    else
                        echo "✗ $(basename $collection_file) checksum mismatch"
                        return 1
                    fi
                fi
            done
            ;;
        "config")
            # Verify configuration files are readable
            find "$backup_dir" -name "*.yml" -o -name "*.yaml" -o -name "*.json" | while read file; do
                if ! python -m json.tool "$file" > /dev/null 2>&1 && ! yq eval "$file" > /dev/null 2>&1; then
                    echo "✗ Invalid configuration file: $file"
                    return 1
                fi
            done
            ;;
    esac
    
    echo "✓ Backup verification completed successfully"
    return 0
}

# Verify latest backups
verify_backup "/backups/qdrant/full/$(ls /backups/qdrant/full | tail -1)" "qdrant"
verify_backup "/backups/config/$(ls /backups/config | tail -1)" "config"
```

## Backup Retention Policies

### Retention Schedule
- **Hourly backups**: Keep for 48 hours
- **Daily backups**: Keep for 30 days
- **Weekly backups**: Keep for 12 weeks  
- **Monthly backups**: Keep for 12 months
- **Yearly backups**: Keep for 7 years

### Automated Cleanup
```bash
#!/bin/bash
# Automated backup cleanup based on retention policy

cleanup_old_backups() {
    local backup_type=$1
    local retention_days=$2
    local backup_base_dir="/backups/$backup_type"
    
    echo "Cleaning up $backup_type backups older than $retention_days days"
    
    find "$backup_base_dir" -type d -mtime +$retention_days -exec rm -rf {} \;
    
    echo "Cleanup completed for $backup_type"
}

# Apply retention policies
cleanup_old_backups "qdrant/incremental" 2    # 48 hours
cleanup_old_backups "qdrant/full" 30          # 30 days
cleanup_old_backups "config" 30               # 30 days
cleanup_old_backups "userdata" 30             # 30 days

# Archive monthly backups to long-term storage
archive_monthly_backups() {
    local month_dir="/backups/archive/$(date +%Y/%m)"
    mkdir -p "$month_dir"
    
    # Move first backup of month to archive
    first_backup=$(find /backups/qdrant/full -name "$(date +%Y%m)01_*" | head -1)
    if [ -n "$first_backup" ]; then
        mv "$first_backup" "$month_dir/"
        echo "Monthly backup archived: $month_dir"
    fi
}

archive_monthly_backups
```

## Backup Security

### Encryption at Rest
```bash
#!/bin/bash
# Encrypt backups using GPG

encrypt_backup() {
    local backup_dir=$1
    local encryption_key="backup-encryption@company.com"
    
    # Create encrypted archive
    tar czf - -C "$backup_dir" . | \
        gpg --trust-model always --encrypt -r "$encryption_key" \
        > "${backup_dir}.tar.gz.gpg"
    
    # Verify encryption
    if [ $? -eq 0 ]; then
        echo "✓ Backup encrypted successfully: ${backup_dir}.tar.gz.gpg"
        rm -rf "$backup_dir"  # Remove unencrypted backup
    else
        echo "✗ Backup encryption failed"
        return 1
    fi
}

# Encrypt all backup types
for backup_type in qdrant config userdata; do
    latest_backup=$(ls -t /backups/$backup_type | head -1)
    if [ -n "$latest_backup" ]; then
        encrypt_backup "/backups/$backup_type/$latest_backup"
    fi
done
```

### Access Control
```bash
#!/bin/bash
# Set proper backup file permissions

# Backup directory permissions
chmod 750 /backups
chown root:backup /backups

# Individual backup permissions
find /backups -type f -exec chmod 640 {} \;
find /backups -type d -exec chmod 750 {} \;

# Restrict access to backup scripts
chmod 750 /opt/backup-scripts/*
chown root:backup /opt/backup-scripts/*

echo "Backup security permissions applied"
```

## Monitoring Integration

### Backup Metrics Collection
```bash
#!/bin/bash
# Export backup metrics to Prometheus

METRICS_FILE="/var/lib/node_exporter/textfile_collector/backup_metrics.prom"

export_backup_metrics() {
    {
        echo "# HELP backup_success Indicates if the last backup was successful"
        echo "# TYPE backup_success gauge"
        echo "backup_success{type=\"qdrant_full\"} $([[ -f /var/lib/backup/last-full-success ]] && echo 1 || echo 0)"
        echo "backup_success{type=\"qdrant_incremental\"} $([[ -f /var/lib/backup/last-incremental-success ]] && echo 1 || echo 0)"
        
        echo "# HELP backup_duration_seconds Duration of the last backup in seconds"
        echo "# TYPE backup_duration_seconds gauge"
        if [ -f /var/lib/backup/last-full-duration ]; then
            echo "backup_duration_seconds{type=\"qdrant_full\"} $(cat /var/lib/backup/last-full-duration)"
        fi
        
        echo "# HELP backup_size_bytes Size of the latest backup in bytes"
        echo "# TYPE backup_size_bytes gauge"
        latest_backup=$(ls -t /backups/qdrant/full | head -1)
        if [ -n "$latest_backup" ]; then
            size=$(du -sb "/backups/qdrant/full/$latest_backup" | cut -f1)
            echo "backup_size_bytes{type=\"qdrant_full\"} $size"
        fi
        
        echo "# HELP backup_age_seconds Age of the latest backup in seconds"
        echo "# TYPE backup_age_seconds gauge"
        if [ -n "$latest_backup" ]; then
            backup_timestamp=$(stat -c %Y "/backups/qdrant/full/$latest_backup")
            current_timestamp=$(date +%s)
            age=$((current_timestamp - backup_timestamp))
            echo "backup_age_seconds{type=\"qdrant_full\"} $age"
        fi
    } > "$METRICS_FILE"
}

export_backup_metrics
```

---

This backup strategy ensures comprehensive data protection with multiple recovery options, automated verification, and proper monitoring integration. Regular testing and validation procedures ensure backup reliability when disaster recovery is needed.