#!/bin/bash

# Qdrant-specific Backup Script for MCP Workspace
# Optimized backup for Qdrant vector database collections
# Usage: ./qdrant-backup.sh [options] [collection_name...]

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKUP_BASE_DIR="${BACKUP_BASE_DIR:-/backups}"
LOG_FILE="${LOG_FILE:-/var/log/backup/qdrant-backup.log}"
BACKUP_DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="${BACKUP_BASE_DIR}/qdrant/${BACKUP_DATE}"

# Default settings
QDRANT_URL="${QDRANT_URL:-http://localhost:6333}"
BACKUP_TYPE="incremental"  # full, incremental, or selective
DRY_RUN=false
VERIFY_INTEGRITY=true
PARALLEL_JOBS=3
RETENTION_DAYS=30

# State tracking
STATE_DIR="/var/lib/backup/qdrant-state"
LAST_BACKUP_TIMESTAMP_FILE="$STATE_DIR/last-backup-timestamp"
COLLECTION_STATES_DIR="$STATE_DIR/collection-states"

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
        log "ERROR" "Qdrant backup failed with exit code: $exit_code"
        if [ -d "$BACKUP_DIR" ] && [ "$DRY_RUN" = false ]; then
            log "INFO" "Cleaning up incomplete backup directory: $BACKUP_DIR"
            rm -rf "$BACKUP_DIR"
        fi
        update_metrics 0 "$START_TIME" "$(date +%s)"
    fi
    exit $exit_code
}

trap cleanup EXIT

# Parse command line arguments
SPECIFIC_COLLECTIONS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --type)
            BACKUP_TYPE="$2"
            shift 2
            ;;
        --url)
            QDRANT_URL="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --no-verify)
            VERIFY_INTEGRITY=false
            shift
            ;;
        --parallel-jobs)
            PARALLEL_JOBS="$2"
            shift 2
            ;;
        --retention-days)
            RETENTION_DAYS="$2"
            shift 2
            ;;
        --collection)
            SPECIFIC_COLLECTIONS+=("$2")
            BACKUP_TYPE="selective"
            shift 2
            ;;
        -h|--help)
            cat << EOF
Usage: $0 [options] [collection_name...]

Options:
  --type TYPE          Backup type: full, incremental, selective (default: incremental)
  --url URL            Qdrant server URL (default: http://localhost:6333)
  --dry-run            Show what would be done without executing
  --no-verify          Skip backup integrity verification
  --parallel-jobs N    Number of parallel backup jobs (default: 3)
  --retention-days N   Backup retention in days (default: 30)
  --collection NAME    Specific collection to backup (can be repeated)
  -h, --help           Show this help message

Backup Types:
  full         - Backup all collections regardless of changes
  incremental  - Only backup collections that have changed since last backup
  selective    - Backup only specified collections

Examples:
  $0                                    # Incremental backup of all collections
  $0 --type full                        # Full backup of all collections
  $0 --collection user-docs             # Backup only user-docs collection
  $0 --collection user-docs --collection embeddings  # Backup multiple specific collections
EOF
            exit 0
            ;;
        -*)
            log "ERROR" "Unknown parameter: $1"
            exit 1
            ;;
        *)
            SPECIFIC_COLLECTIONS+=("$1")
            BACKUP_TYPE="selective"
            shift
            ;;
    esac
done

# Initialize state tracking
init_state_tracking() {
    mkdir -p "$STATE_DIR" "$COLLECTION_STATES_DIR"
    if [ ! -f "$LAST_BACKUP_TIMESTAMP_FILE" ]; then
        echo "0" > "$LAST_BACKUP_TIMESTAMP_FILE"
    fi
}

# Check Qdrant connectivity and health
check_qdrant_health() {
    log "INFO" "Checking Qdrant connectivity at $QDRANT_URL"
    
    if ! curl -s --max-time 10 "${QDRANT_URL}/healthz" > /dev/null; then
        log "ERROR" "Qdrant service is not accessible at $QDRANT_URL"
        return 1
    fi
    
    # Get Qdrant version and status
    local qdrant_info
    if qdrant_info=$(curl -s --max-time 5 "${QDRANT_URL}/"); then
        local version
        version=$(echo "$qdrant_info" | jq -r '.version // "unknown"')
        log "INFO" "Connected to Qdrant version: $version"
    else
        log "WARN" "Could not retrieve Qdrant version information"
    fi
    
    return 0
}

# Get list of collections to backup
get_collections_to_backup() {
    local collections_json
    collections_json=$(curl -s "${QDRANT_URL}/collections" | jq -r '.result.collections[]')
    
    if [ -z "$collections_json" ]; then
        log "WARN" "No collections found in Qdrant"
        return 0
    fi
    
    case $BACKUP_TYPE in
        "full")
            echo "$collections_json" | jq -r '.name'
            ;;
        "incremental")
            get_changed_collections
            ;;
        "selective")
            if [ ${#SPECIFIC_COLLECTIONS[@]} -eq 0 ]; then
                log "ERROR" "No collections specified for selective backup"
                return 1
            fi
            printf '%s\n' "${SPECIFIC_COLLECTIONS[@]}"
            ;;
    esac
}

# Identify collections that have changed since last backup
get_changed_collections() {
    local last_backup_timestamp
    last_backup_timestamp=$(cat "$LAST_BACKUP_TIMESTAMP_FILE")
    local changed_collections=()
    
    # Get all collections info
    local collections_info
    collections_info=$(curl -s "${QDRANT_URL}/collections" | jq -r '.result.collections[]')
    
    echo "$collections_info" | jq -r '.name' | while read collection; do
        # Get current collection info
        local collection_info
        collection_info=$(curl -s "${QDRANT_URL}/collections/${collection}")
        
        if [ $? -ne 0 ]; then
            log "WARN" "Could not get info for collection: $collection"
            continue
        fi
        
        local points_count
        local vectors_count
        points_count=$(echo "$collection_info" | jq -r '.result.points_count // 0')
        vectors_count=$(echo "$collection_info" | jq -r '.result.vectors_count // 0')
        
        # Check if collection state has changed
        local state_file="$COLLECTION_STATES_DIR/${collection}.state"
        local current_state="${points_count}:${vectors_count}"
        
        if [ ! -f "$state_file" ] || [ "$(cat "$state_file")" != "$current_state" ]; then
            echo "$collection"
            log "INFO" "Collection $collection has changed: $current_state"
        else
            log "DEBUG" "Collection $collection unchanged, skipping"
        fi
    done
}

# Create collection snapshot
create_collection_snapshot() {
    local collection=$1
    local max_retries=3
    local retry_count=0
    
    while [ $retry_count -lt $max_retries ]; do
        log "INFO" "Creating snapshot for collection: $collection (attempt $((retry_count + 1)))"
        
        local snapshot_response
        snapshot_response=$(curl -s -X POST "${QDRANT_URL}/collections/${collection}/snapshots")
        
        if echo "$snapshot_response" | jq -e '.result.name' > /dev/null; then
            local snapshot_name
            snapshot_name=$(echo "$snapshot_response" | jq -r '.result.name')
            echo "$snapshot_name"
            return 0
        else
            log "WARN" "Failed to create snapshot for $collection (attempt $((retry_count + 1)))"
            retry_count=$((retry_count + 1))
            sleep 5
        fi
    done
    
    log "ERROR" "Failed to create snapshot for $collection after $max_retries attempts"
    return 1
}

# Download and verify collection snapshot
download_collection_snapshot() {
    local collection=$1
    local snapshot_name=$2
    local output_file="$3"
    
    log "INFO" "Downloading snapshot for collection: $collection"
    
    local start_time
    start_time=$(date +%s)
    
    if ! curl -s "${QDRANT_URL}/collections/${collection}/snapshots/${snapshot_name}" \
         --output "$output_file"; then
        log "ERROR" "Failed to download snapshot for collection: $collection"
        return 1
    fi
    
    local end_time
    end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local file_size
    file_size=$(stat -f%z "$output_file" 2>/dev/null || stat -c%s "$output_file")
    
    log "INFO" "Downloaded $collection snapshot in ${duration}s ($(numfmt --to=iec $file_size))"
    
    # Verify file integrity
    if [ "$VERIFY_INTEGRITY" = true ]; then
        if [ ! -s "$output_file" ]; then
            log "ERROR" "Downloaded snapshot for $collection is empty"
            return 1
        fi
        
        # Basic format validation (Qdrant snapshots are typically tar.gz files)
        if ! file "$output_file" | grep -q "gzip"; then
            log "WARN" "Snapshot for $collection may not be in expected gzip format"
        fi
    fi
    
    return 0
}

# Backup single collection
backup_collection() {
    local collection=$1
    local start_time
    start_time=$(date +%s)
    
    log "INFO" "Starting backup for collection: $collection"
    
    if [ "$DRY_RUN" = true ]; then
        log "INFO" "[DRY-RUN] Would backup collection: $collection"
        return 0
    fi
    
    # Get collection info before backup
    local collection_info
    collection_info=$(curl -s "${QDRANT_URL}/collections/${collection}")
    
    if [ $? -ne 0 ] || ! echo "$collection_info" | jq -e '.result' > /dev/null; then
        log "ERROR" "Collection $collection does not exist or is not accessible"
        return 1
    fi
    
    # Extract collection stats
    local points_count vectors_count status
    points_count=$(echo "$collection_info" | jq -r '.result.points_count // 0')
    vectors_count=$(echo "$collection_info" | jq -r '.result.vectors_count // 0')
    status=$(echo "$collection_info" | jq -r '.result.status // "unknown"')
    
    if [ "$status" != "green" ]; then
        log "WARN" "Collection $collection status is $status, backup may be inconsistent"
    fi
    
    # Create snapshot
    local snapshot_name
    if ! snapshot_name=$(create_collection_snapshot "$collection"); then
        echo "$collection" >> "$BACKUP_DIR/failed_collections.txt"
        return 1
    fi
    
    # Download snapshot
    local output_file="$BACKUP_DIR/${collection}.snapshot"
    if ! download_collection_snapshot "$collection" "$snapshot_name" "$output_file"; then
        echo "$collection" >> "$BACKUP_DIR/failed_collections.txt"
        return 1
    fi
    
    # Generate collection metadata
    local checksum
    checksum=$(sha256sum "$output_file" | cut -d' ' -f1)
    local file_size
    file_size=$(stat -f%z "$output_file" 2>/dev/null || stat -c%s "$output_file")
    
    cat > "$BACKUP_DIR/${collection}.meta" << EOF
{
  "collection_name": "${collection}",
  "snapshot_name": "${snapshot_name}",
  "backup_date": "${BACKUP_DATE}",
  "backup_timestamp": $(date +%s),
  "backup_type": "${BACKUP_TYPE}",
  "file_size": ${file_size},
  "checksum": "${checksum}",
  "collection_stats": {
    "points_count": ${points_count},
    "vectors_count": ${vectors_count},
    "status": "${status}"
  },
  "collection_config": $(echo "$collection_info" | jq '.result.config // {}')
}
EOF
    
    # Update collection state
    local current_state="${points_count}:${vectors_count}"
    echo "$current_state" > "$COLLECTION_STATES_DIR/${collection}.state"
    
    # Clean up remote snapshot
    curl -s -X DELETE "${QDRANT_URL}/collections/${collection}/snapshots/${snapshot_name}" > /dev/null || true
    
    local end_time
    end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    log "INFO" "✓ Collection $collection backed up successfully in ${duration}s"
    return 0
}

# Verify backup integrity
verify_backup_integrity() {
    if [ "$VERIFY_INTEGRITY" = false ]; then
        log "INFO" "Skipping backup integrity verification (disabled)"
        return 0
    fi
    
    log "INFO" "Verifying backup integrity..."
    
    local failed_verifications=0
    
    for meta_file in "$BACKUP_DIR"/*.meta; do
        if [ ! -f "$meta_file" ]; then
            continue
        fi
        
        local collection_name
        collection_name=$(jq -r '.collection_name' "$meta_file")
        local expected_checksum
        expected_checksum=$(jq -r '.checksum' "$meta_file")
        local snapshot_file="$BACKUP_DIR/${collection_name}.snapshot"
        
        if [ ! -f "$snapshot_file" ]; then
            log "ERROR" "Snapshot file missing for collection: $collection_name"
            failed_verifications=$((failed_verifications + 1))
            continue
        fi
        
        local actual_checksum
        actual_checksum=$(sha256sum "$snapshot_file" | cut -d' ' -f1)
        
        if [ "$expected_checksum" = "$actual_checksum" ]; then
            log "INFO" "✓ Integrity verified for collection: $collection_name"
        else
            log "ERROR" "✗ Integrity check failed for collection: $collection_name"
            failed_verifications=$((failed_verifications + 1))
        fi
    done
    
    if [ $failed_verifications -gt 0 ]; then
        log "ERROR" "$failed_verifications collections failed integrity verification"
        return 1
    else
        log "INFO" "All backup integrity checks passed"
        return 0
    fi
}

# Create backup manifest
create_backup_manifest() {
    log "INFO" "Creating backup manifest..."
    
    if [ "$DRY_RUN" = true ]; then
        log "INFO" "[DRY-RUN] Would create backup manifest"
        return 0
    fi
    
    local qdrant_version
    qdrant_version=$(curl -s "${QDRANT_URL}/" | jq -r '.version' 2>/dev/null || echo "unknown")
    
    local collection_count=0
    local total_size=0
    local failed_count=0
    
    if [ -d "$BACKUP_DIR" ]; then
        collection_count=$(find "$BACKUP_DIR" -name "*.snapshot" | wc -l)
        total_size=$(du -sb "$BACKUP_DIR" | cut -f1)
    fi
    
    if [ -f "$BACKUP_DIR/failed_collections.txt" ]; then
        failed_count=$(wc -l < "$BACKUP_DIR/failed_collections.txt")
    fi
    
    cat > "$BACKUP_DIR/manifest.json" << EOF
{
  "backup_type": "${BACKUP_TYPE}",
  "backup_date": "${BACKUP_DATE}",
  "backup_timestamp": $(date +%s),
  "qdrant_url": "${QDRANT_URL}",
  "qdrant_version": "${qdrant_version}",
  "collections_backed_up": ${collection_count},
  "collections_failed": ${failed_count},
  "total_size_bytes": ${total_size},
  "verify_integrity": ${VERIFY_INTEGRITY},
  "parallel_jobs": ${PARALLEL_JOBS},
  "created_by": "qdrant-backup-script",
  "hostname": "$(hostname)",
  "backup_duration_seconds": 0
}
EOF
    
    log "INFO" "Backup manifest created"
}

# Clean up old backups
cleanup_old_backups() {
    log "INFO" "Cleaning up backups older than $RETENTION_DAYS days..."
    
    if [ "$DRY_RUN" = true ]; then
        log "INFO" "[DRY-RUN] Would clean up old backups"
        return 0
    fi
    
    local backup_parent_dir="$(dirname "$BACKUP_DIR")"
    local deleted_count=0
    
    find "$backup_parent_dir" -maxdepth 1 -type d -mtime +$RETENTION_DAYS | while read old_backup; do
        if [ "$old_backup" != "$backup_parent_dir" ]; then
            log "INFO" "Removing old backup: $(basename "$old_backup")"
            rm -rf "$old_backup"
            deleted_count=$((deleted_count + 1))
        fi
    done
    
    if [ $deleted_count -gt 0 ]; then
        log "INFO" "Cleaned up $deleted_count old backups"
    else
        log "INFO" "No old backups to clean up"
    fi
}

# Update backup metrics for Prometheus
update_metrics() {
    local success=$1
    local start_time=$2
    local end_time=$3
    
    local metrics_file="/var/lib/node_exporter/textfile_collector/qdrant_backup_metrics.prom"
    local duration=$((end_time - start_time))
    local backup_size=0
    
    if [ -d "$BACKUP_DIR" ] && [ "$DRY_RUN" = false ]; then
        backup_size=$(du -sb "$BACKUP_DIR" | cut -f1)
    fi
    
    local collection_count=0
    if [ -d "$BACKUP_DIR" ]; then
        collection_count=$(find "$BACKUP_DIR" -name "*.snapshot" 2>/dev/null | wc -l)
    fi
    
    mkdir -p "$(dirname "$metrics_file")"
    cat > "$metrics_file" << EOF
# HELP qdrant_backup_success Indicates if the last Qdrant backup was successful
# TYPE qdrant_backup_success gauge
qdrant_backup_success{type="${BACKUP_TYPE}"} $success

# HELP qdrant_backup_duration_seconds Duration of the last Qdrant backup in seconds
# TYPE qdrant_backup_duration_seconds gauge
qdrant_backup_duration_seconds{type="${BACKUP_TYPE}"} $duration

# HELP qdrant_backup_size_bytes Size of the latest Qdrant backup in bytes
# TYPE qdrant_backup_size_bytes gauge
qdrant_backup_size_bytes{type="${BACKUP_TYPE}"} $backup_size

# HELP qdrant_backup_collections_count Number of collections in the latest backup
# TYPE qdrant_backup_collections_count gauge
qdrant_backup_collections_count{type="${BACKUP_TYPE}"} $collection_count

# HELP qdrant_backup_timestamp_seconds Unix timestamp of the last Qdrant backup
# TYPE qdrant_backup_timestamp_seconds gauge
qdrant_backup_timestamp_seconds{type="${BACKUP_TYPE}"} $end_time
EOF
    
    log "INFO" "Backup metrics updated"
}

# Main backup function
main() {
    local START_TIME
    START_TIME=$(date +%s)
    
    log "INFO" "Starting Qdrant backup (type: $BACKUP_TYPE, dry-run: $DRY_RUN)"
    log "INFO" "Qdrant URL: $QDRANT_URL"
    log "INFO" "Backup directory: $BACKUP_DIR"
    
    # Initialize
    mkdir -p "$(dirname "$LOG_FILE")"
    mkdir -p "/var/lib/backup"
    init_state_tracking
    check_qdrant_health
    
    if [ "$DRY_RUN" = false ]; then
        mkdir -p "$BACKUP_DIR"
        chmod 750 "$BACKUP_DIR"
    fi
    
    # Get collections to backup
    local collections
    collections=$(get_collections_to_backup)
    
    if [ -z "$collections" ]; then
        log "INFO" "No collections to backup"
        update_metrics 1 "$START_TIME" "$(date +%s)"
        return 0
    fi
    
    local collection_count
    collection_count=$(echo "$collections" | wc -l)
    log "INFO" "Found $collection_count collections to backup: $(echo "$collections" | tr '\n' ', ' | sed 's/,$//')"
    
    # Backup collections in parallel
    export -f backup_collection create_collection_snapshot download_collection_snapshot log
    export BACKUP_DIR BACKUP_DATE BACKUP_TYPE DRY_RUN VERIFY_INTEGRITY QDRANT_URL COLLECTION_STATES_DIR
    
    echo "$collections" | xargs -n 1 -P "$PARALLEL_JOBS" -I {} bash -c 'backup_collection "$@"' _ {}
    
    # Post-backup tasks
    create_backup_manifest
    verify_backup_integrity
    cleanup_old_backups
    
    # Update state
    if [ "$DRY_RUN" = false ]; then
        echo "$(date +%s)" > "$LAST_BACKUP_TIMESTAMP_FILE"
        
        # Update manifest with actual duration
        local end_time
        end_time=$(date +%s)
        local duration=$((end_time - START_TIME))
        
        if [ -f "$BACKUP_DIR/manifest.json" ]; then
            jq --argjson duration "$duration" '.backup_duration_seconds = $duration' \
               "$BACKUP_DIR/manifest.json" > "${BACKUP_DIR}/manifest.tmp" && \
               mv "${BACKUP_DIR}/manifest.tmp" "$BACKUP_DIR/manifest.json"
        fi
    fi
    
    local end_time
    end_time=$(date +%s)
    local duration=$((end_time - START_TIME))
    
    # Check for failures
    local success=1
    if [ -f "$BACKUP_DIR/failed_collections.txt" ]; then
        local failed_count
        failed_count=$(wc -l < "$BACKUP_DIR/failed_collections.txt")
        log "WARN" "$failed_count collections failed to backup"
        success=0
    fi
    
    update_metrics $success "$START_TIME" "$end_time"
    
    if [ $success -eq 1 ]; then
        log "INFO" "Qdrant backup completed successfully in ${duration}s"
        echo "Backup location: $BACKUP_DIR"
        echo "Collections backed up: $collection_count"
    else
        log "ERROR" "Qdrant backup completed with failures"
        exit 1
    fi
}

# Run main function
main "$@"