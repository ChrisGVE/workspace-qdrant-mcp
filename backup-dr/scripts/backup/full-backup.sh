#!/bin/bash

# Full System Backup Script for Qdrant MCP Workspace
# Performs comprehensive backup of all critical components
# Usage: ./full-backup.sh [--dry-run] [--compress] [--upload]

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKUP_BASE_DIR="${BACKUP_BASE_DIR:-/backups}"
LOG_FILE="${LOG_FILE:-/var/log/backup/full-backup.log}"
BACKUP_DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="${BACKUP_BASE_DIR}/full/${BACKUP_DATE}"

# Default settings
DRY_RUN=false
COMPRESS=true
UPLOAD=true
PARALLEL_JOBS=4

# Logging function
log() {
    local level=$1
    shift
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$level] $*" | tee -a "$LOG_FILE"
}

# Error handling
cleanup() {
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        log "ERROR" "Backup failed with exit code: $exit_code"
        # Clean up incomplete backup
        if [ -d "$BACKUP_DIR" ] && [ "$DRY_RUN" = false ]; then
            log "INFO" "Cleaning up incomplete backup directory: $BACKUP_DIR"
            rm -rf "$BACKUP_DIR"
        fi
    fi
    exit $exit_code
}

trap cleanup EXIT

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --no-compress)
            COMPRESS=false
            shift
            ;;
        --no-upload)
            UPLOAD=false
            shift
            ;;
        --parallel-jobs)
            PARALLEL_JOBS="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--dry-run] [--no-compress] [--no-upload] [--parallel-jobs N]"
            echo "  --dry-run        Show what would be done without executing"
            echo "  --no-compress    Skip compression of backup files"
            echo "  --no-upload      Skip upload to remote storage"
            echo "  --parallel-jobs  Number of parallel backup jobs (default: 4)"
            exit 0
            ;;
        *)
            log "ERROR" "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

# Pre-backup checks
check_prerequisites() {
    log "INFO" "Checking prerequisites..."
    
    # Check if Qdrant is accessible
    if ! curl -s --max-time 10 "http://localhost:6333/healthz" > /dev/null; then
        log "ERROR" "Qdrant service is not accessible"
        return 1
    fi
    
    # Check disk space (need at least 10GB free)
    local available_space
    available_space=$(df "$BACKUP_BASE_DIR" | awk 'NR==2 {print $4}')
    if [ "$available_space" -lt 10485760 ]; then  # 10GB in KB
        log "ERROR" "Insufficient disk space. Available: ${available_space}KB, Required: 10GB"
        return 1
    fi
    
    # Check required tools
    local required_tools=("curl" "jq" "tar" "gzip")
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            log "ERROR" "Required tool not found: $tool"
            return 1
        fi
    done
    
    log "INFO" "Prerequisites check passed"
}

# Create backup directory structure
setup_backup_dir() {
    log "INFO" "Setting up backup directory: $BACKUP_DIR"
    
    if [ "$DRY_RUN" = false ]; then
        mkdir -p "$BACKUP_DIR"/{qdrant,config,data,logs,metadata}
        chmod 750 "$BACKUP_DIR"
    fi
}

# Backup Qdrant collections
backup_qdrant() {
    log "INFO" "Starting Qdrant backup..."
    
    local qdrant_backup_dir="$BACKUP_DIR/qdrant"
    local failed_collections=()
    
    # Get list of all collections
    local collections
    collections=$(curl -s "http://localhost:6333/collections" | jq -r '.result.collections[].name')
    
    if [ -z "$collections" ]; then
        log "WARN" "No Qdrant collections found"
        return 0
    fi
    
    log "INFO" "Found $(echo "$collections" | wc -l) collections to backup"
    
    # Backup collections in parallel
    backup_collection() {
        local collection=$1
        local start_time
        start_time=$(date +%s)
        
        log "INFO" "Backing up collection: $collection"
        
        if [ "$DRY_RUN" = true ]; then
            log "INFO" "[DRY-RUN] Would backup collection: $collection"
            return 0
        fi
        
        # Create collection snapshot
        local snapshot_response
        snapshot_response=$(curl -s -X POST "http://localhost:6333/collections/${collection}/snapshots")
        
        if ! echo "$snapshot_response" | jq -e '.result.name' > /dev/null; then
            log "ERROR" "Failed to create snapshot for collection: $collection"
            echo "$collection" >> "$BACKUP_DIR/failed_collections.txt"
            return 1
        fi
        
        local snapshot_name
        snapshot_name=$(echo "$snapshot_response" | jq -r '.result.name')
        
        # Download snapshot
        if ! curl -s "http://localhost:6333/collections/${collection}/snapshots/${snapshot_name}" \
             --output "${qdrant_backup_dir}/${collection}.snapshot"; then
            log "ERROR" "Failed to download snapshot for collection: $collection"
            echo "$collection" >> "$BACKUP_DIR/failed_collections.txt"
            return 1
        fi
        
        # Get collection info for metadata
        local collection_info
        collection_info=$(curl -s "http://localhost:6333/collections/$collection")
        
        # Generate metadata
        cat > "${qdrant_backup_dir}/${collection}.meta" << EOF
{
  "collection_name": "${collection}",
  "snapshot_name": "${snapshot_name}",
  "backup_date": "${BACKUP_DATE}",
  "file_size": $(stat -f%z "${qdrant_backup_dir}/${collection}.snapshot" 2>/dev/null || stat -c%s "${qdrant_backup_dir}/${collection}.snapshot"),
  "checksum": "$(sha256sum "${qdrant_backup_dir}/${collection}.snapshot" | cut -d' ' -f1)",
  "collection_info": $collection_info
}
EOF
        
        local end_time
        end_time=$(date +%s)
        local duration=$((end_time - start_time))
        
        log "INFO" "✓ Collection $collection backed up in ${duration}s"
        
        # Clean up snapshot from Qdrant
        curl -s -X DELETE "http://localhost:6333/collections/${collection}/snapshots/${snapshot_name}" > /dev/null
    }
    
    export -f backup_collection log
    export BACKUP_DIR BACKUP_DATE DRY_RUN LOG_FILE
    
    # Run backups in parallel
    echo "$collections" | xargs -n 1 -P "$PARALLEL_JOBS" -I {} bash -c 'backup_collection "$@"' _ {}
    
    # Check for failed collections
    if [ -f "$BACKUP_DIR/failed_collections.txt" ]; then
        local failed_count
        failed_count=$(wc -l < "$BACKUP_DIR/failed_collections.txt")
        log "WARN" "$failed_count collections failed to backup"
        log "WARN" "Failed collections: $(cat "$BACKUP_DIR/failed_collections.txt" | tr '\n' ', ')"
    else
        log "INFO" "All Qdrant collections backed up successfully"
    fi
}

# Backup application configuration
backup_configuration() {
    log "INFO" "Starting configuration backup..."
    
    local config_backup_dir="$BACKUP_DIR/config"
    
    if [ "$DRY_RUN" = true ]; then
        log "INFO" "[DRY-RUN] Would backup configuration files"
        return 0
    fi
    
    # Backup environment files
    if [ -f ".env" ]; then
        cp .env "$config_backup_dir/env" 2>/dev/null || true
    fi
    
    # Backup Docker configurations
    find . -maxdepth 1 -name "docker-compose*.yml" -exec cp {} "$config_backup_dir/" \; 2>/dev/null || true
    
    # Backup MCP configuration
    if [ -f ".mcp.json" ]; then
        cp .mcp.json "$config_backup_dir/" 2>/dev/null || true
    fi
    
    # Backup monitoring configuration
    if [ -d "monitoring" ]; then
        cp -r monitoring "$config_backup_dir/" 2>/dev/null || true
    fi
    
    # Backup Kubernetes manifests (if available)
    if command -v kubectl &> /dev/null; then
        kubectl get configmaps -o yaml > "$config_backup_dir/k8s-configmaps.yaml" 2>/dev/null || true
        kubectl get secrets -o yaml > "$config_backup_dir/k8s-secrets.yaml" 2>/dev/null || true
    fi
    
    # Backup systemd service files
    if [ -d "/etc/systemd/system" ]; then
        find /etc/systemd/system -name "*qdrant*" -o -name "*mcp*" | \
            xargs -I {} cp {} "$config_backup_dir/" 2>/dev/null || true
    fi
    
    log "INFO" "Configuration backup completed"
}

# Backup application data
backup_data() {
    log "INFO" "Starting application data backup..."
    
    local data_backup_dir="$BACKUP_DIR/data"
    
    if [ "$DRY_RUN" = true ]; then
        log "INFO" "[DRY-RUN] Would backup application data"
        return 0
    fi
    
    # Backup user uploads (if exists)
    if [ -d "/app/uploads" ]; then
        rsync -av /app/uploads/ "$data_backup_dir/uploads/" 2>/dev/null || true
    fi
    
    # Backup application cache (if exists)
    if [ -d "/app/cache" ]; then
        cp -r /app/cache "$data_backup_dir/" 2>/dev/null || true
    fi
    
    # Backup user sessions (if file-based)
    if [ -d "/app/sessions" ]; then
        cp -r /app/sessions "$data_backup_dir/" 2>/dev/null || true
    fi
    
    # Backup application state
    if [ -d "/var/lib/qdrant-mcp" ]; then
        cp -r /var/lib/qdrant-mcp "$data_backup_dir/app-state/" 2>/dev/null || true
    fi
    
    log "INFO" "Application data backup completed"
}

# Backup logs
backup_logs() {
    log "INFO" "Starting logs backup..."
    
    local logs_backup_dir="$BACKUP_DIR/logs"
    
    if [ "$DRY_RUN" = true ]; then
        log "INFO" "[DRY-RUN] Would backup logs"
        return 0
    fi
    
    # Application logs
    if command -v journalctl &> /dev/null; then
        journalctl --unit=qdrant-mcp --since="24 hours ago" --no-pager > "$logs_backup_dir/application.log" 2>/dev/null || true
    fi
    
    # Docker logs (if using Docker)
    if command -v docker &> /dev/null; then
        docker ps --format "{{.Names}}" | grep -E "(qdrant|mcp)" | while read container; do
            docker logs --since=24h "$container" > "$logs_backup_dir/${container}.log" 2>&1 || true
        done
    fi
    
    # System logs
    if [ -d "/var/log" ]; then
        find /var/log -name "*qdrant*" -o -name "*mcp*" -type f -mtime -1 | \
            xargs -I {} cp {} "$logs_backup_dir/" 2>/dev/null || true
    fi
    
    log "INFO" "Logs backup completed"
}

# Create backup manifest
create_manifest() {
    log "INFO" "Creating backup manifest..."
    
    if [ "$DRY_RUN" = true ]; then
        log "INFO" "[DRY-RUN] Would create backup manifest"
        return 0
    fi
    
    local qdrant_version
    qdrant_version=$(curl -s "http://localhost:6333/" | jq -r '.version' 2>/dev/null || echo "unknown")
    
    local collection_count=0
    if [ -d "$BACKUP_DIR/qdrant" ]; then
        collection_count=$(find "$BACKUP_DIR/qdrant" -name "*.snapshot" | wc -l)
    fi
    
    local total_size=0
    if [ -d "$BACKUP_DIR" ]; then
        total_size=$(du -sb "$BACKUP_DIR" | cut -f1)
    fi
    
    cat > "$BACKUP_DIR/manifest.json" << EOF
{
  "backup_type": "full",
  "backup_date": "${BACKUP_DATE}",
  "backup_timestamp": $(date +%s),
  "qdrant_version": "${qdrant_version}",
  "collections_count": ${collection_count},
  "total_size_bytes": ${total_size},
  "components": {
    "qdrant_collections": $([ -d "$BACKUP_DIR/qdrant" ] && echo true || echo false),
    "configuration": $([ -d "$BACKUP_DIR/config" ] && echo true || echo false),
    "application_data": $([ -d "$BACKUP_DIR/data" ] && echo true || echo false),
    "logs": $([ -d "$BACKUP_DIR/logs" ] && echo true || echo false)
  },
  "backup_duration_seconds": 0,
  "created_by": "automated-full-backup",
  "hostname": "$(hostname)",
  "git_commit": "$(git rev-parse HEAD 2>/dev/null || echo 'unknown')"
}
EOF
    
    log "INFO" "Backup manifest created"
}

# Compress backup
compress_backup() {
    if [ "$COMPRESS" = false ]; then
        log "INFO" "Skipping compression (disabled)"
        return 0
    fi
    
    log "INFO" "Compressing backup..."
    
    if [ "$DRY_RUN" = true ]; then
        log "INFO" "[DRY-RUN] Would compress backup to ${BACKUP_DIR}.tar.gz"
        return 0
    fi
    
    local start_time
    start_time=$(date +%s)
    
    # Create compressed archive
    tar -czf "${BACKUP_DIR}.tar.gz" -C "$(dirname "$BACKUP_DIR")" "$(basename "$BACKUP_DIR")"
    
    if [ $? -eq 0 ]; then
        # Remove uncompressed backup
        rm -rf "$BACKUP_DIR"
        
        local end_time
        end_time=$(date +%s)
        local duration=$((end_time - start_time))
        
        local compressed_size
        compressed_size=$(stat -f%z "${BACKUP_DIR}.tar.gz" 2>/dev/null || stat -c%s "${BACKUP_DIR}.tar.gz")
        
        log "INFO" "Backup compressed in ${duration}s, size: $(numfmt --to=iec $compressed_size)"
    else
        log "ERROR" "Backup compression failed"
        return 1
    fi
}

# Upload backup to remote storage
upload_backup() {
    if [ "$UPLOAD" = false ]; then
        log "INFO" "Skipping upload (disabled)"
        return 0
    fi
    
    log "INFO" "Uploading backup to remote storage..."
    
    if [ "$DRY_RUN" = true ]; then
        log "INFO" "[DRY-RUN] Would upload backup to remote storage"
        return 0
    fi
    
    local backup_file="${BACKUP_DIR}"
    if [ "$COMPRESS" = true ]; then
        backup_file="${BACKUP_DIR}.tar.gz"
    fi
    
    # Upload to S3 (customize based on your storage provider)
    if command -v aws &> /dev/null; then
        aws s3 cp "$backup_file" "s3://company-backups/qdrant-mcp/full/" || {
            log "ERROR" "Failed to upload backup to S3"
            return 1
        }
        log "INFO" "Backup uploaded to S3 successfully"
    fi
    
    # Upload to additional remote locations if configured
    if [ -n "${BACKUP_RSYNC_TARGET:-}" ]; then
        rsync -av "$backup_file" "$BACKUP_RSYNC_TARGET" || {
            log "ERROR" "Failed to sync backup to $BACKUP_RSYNC_TARGET"
            return 1
        }
        log "INFO" "Backup synced to $BACKUP_RSYNC_TARGET successfully"
    fi
}

# Update backup metrics
update_metrics() {
    log "INFO" "Updating backup metrics..."
    
    local metrics_file="/var/lib/node_exporter/textfile_collector/backup_metrics.prom"
    local success=$1
    local start_time=$2
    local end_time=$3
    
    if [ "$DRY_RUN" = true ]; then
        log "INFO" "[DRY-RUN] Would update backup metrics"
        return 0
    fi
    
    local duration=$((end_time - start_time))
    local backup_size=0
    
    if [ "$COMPRESS" = true ] && [ -f "${BACKUP_DIR}.tar.gz" ]; then
        backup_size=$(stat -f%z "${BACKUP_DIR}.tar.gz" 2>/dev/null || stat -c%s "${BACKUP_DIR}.tar.gz")
    elif [ -d "$BACKUP_DIR" ]; then
        backup_size=$(du -sb "$BACKUP_DIR" | cut -f1)
    fi
    
    # Create metrics file
    mkdir -p "$(dirname "$metrics_file")"
    cat > "$metrics_file" << EOF
# HELP backup_success Indicates if the last backup was successful
# TYPE backup_success gauge
backup_success{type="full"} $success

# HELP backup_duration_seconds Duration of the last backup in seconds
# TYPE backup_duration_seconds gauge
backup_duration_seconds{type="full"} $duration

# HELP backup_size_bytes Size of the latest backup in bytes
# TYPE backup_size_bytes gauge
backup_size_bytes{type="full"} $backup_size

# HELP backup_timestamp_seconds Unix timestamp of the last backup
# TYPE backup_timestamp_seconds gauge
backup_timestamp_seconds{type="full"} $end_time
EOF
    
    # Update success marker files
    if [ "$success" -eq 1 ]; then
        echo "$end_time" > /var/lib/backup/last-full-success
        echo "$duration" > /var/lib/backup/last-full-duration
    fi
    
    log "INFO" "Backup metrics updated"
}

# Main backup function
main() {
    local start_time
    start_time=$(date +%s)
    
    log "INFO" "Starting full system backup (dry-run: $DRY_RUN)"
    log "INFO" "Backup directory: $BACKUP_DIR"
    
    # Setup
    mkdir -p "$(dirname "$LOG_FILE")"
    mkdir -p "/var/lib/backup"
    check_prerequisites
    setup_backup_dir
    
    # Execute backup components
    backup_qdrant
    backup_configuration
    backup_data
    backup_logs
    create_manifest
    
    # Post-processing
    compress_backup
    upload_backup
    
    local end_time
    end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    # Update manifest with actual duration
    if [ "$DRY_RUN" = false ] && [ -f "$BACKUP_DIR/manifest.json" ]; then
        jq --argjson duration "$duration" '.backup_duration_seconds = $duration' "$BACKUP_DIR/manifest.json" > "${BACKUP_DIR}/manifest.tmp" && mv "${BACKUP_DIR}/manifest.tmp" "$BACKUP_DIR/manifest.json"
    fi
    
    update_metrics 1 "$start_time" "$end_time"
    
    log "INFO" "Full backup completed successfully in ${duration}s"
    
    if [ "$DRY_RUN" = false ]; then
        echo "Backup location: $BACKUP_DIR"
        if [ "$COMPRESS" = true ]; then
            echo "Compressed backup: ${BACKUP_DIR}.tar.gz"
        fi
    fi
}

# Run main function
main "$@"