#!/bin/bash

# Emergency Recovery Script for Qdrant MCP Workspace
# Automated disaster recovery from backups with minimal downtime
# Usage: ./emergency-recovery.sh [options]

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKUP_BASE_DIR="${BACKUP_BASE_DIR:-/backups}"
LOG_FILE="${LOG_FILE:-/var/log/recovery/emergency-recovery.log}"
RECOVERY_DATE=$(date +%Y%m%d_%H%M%S)
RECOVERY_DIR="${RECOVERY_DIR:-/tmp/recovery-${RECOVERY_DATE}}"

# Recovery settings
QDRANT_URL="${QDRANT_URL:-http://localhost:6333}"
BACKUP_TYPE="latest"  # latest, specific, point-in-time
TARGET_TIMESTAMP=""
SPECIFIC_BACKUP=""
DRY_RUN=false
SKIP_VERIFICATION=false
PARALLEL_JOBS=4
FORCE_RECOVERY=false

# Recovery tracking
START_TIME=$(date +%s)
RECOVERY_STATUS="in_progress"
FAILED_COLLECTIONS=()
RECOVERED_COLLECTIONS=()

# Logging function
log() {
    local level=$1
    shift
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] [$level] $*" | tee -a "$LOG_FILE"
    
    # Send to incident management system if configured
    if [ -n "${INCIDENT_WEBHOOK_URL:-}" ] && [ "$level" = "ERROR" ]; then
        curl -s -X POST "$INCIDENT_WEBHOOK_URL" \
            -H "Content-Type: application/json" \
            -d "{\"level\":\"$level\",\"message\":\"$*\",\"timestamp\":\"$timestamp\"}" || true
    fi
}

# Emergency alert function
send_alert() {
    local severity=$1
    local message=$2
    
    log "$severity" "$message"
    
    # Send to Slack/Teams if configured
    if [ -n "${EMERGENCY_WEBHOOK_URL:-}" ]; then
        curl -s -X POST "$EMERGENCY_WEBHOOK_URL" \
            -H "Content-Type: application/json" \
            -d "{
                \"text\":\"🚨 EMERGENCY RECOVERY: $message\",
                \"severity\":\"$severity\",
                \"timestamp\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"
            }" || true
    fi
    
    # Update status page if configured
    if [ -n "${STATUS_PAGE_API:-}" ]; then
        curl -s -X POST "$STATUS_PAGE_API/incidents" \
            -H "Authorization: Bearer ${STATUS_PAGE_TOKEN}" \
            -H "Content-Type: application/json" \
            -d "{
                \"name\":\"Emergency Recovery in Progress\",
                \"status\":\"investigating\",
                \"message\":\"$message\"
            }" || true
    fi
}

# Error handling and cleanup
cleanup() {
    local exit_code=$?
    local end_time=$(date +%s)
    local duration=$((end_time - START_TIME))
    
    if [ $exit_code -ne 0 ]; then
        RECOVERY_STATUS="failed"
        log "ERROR" "Emergency recovery failed after ${duration} seconds"
        send_alert "critical" "Emergency recovery failed - manual intervention required"
        
        # Generate failure report
        generate_recovery_report "failed" "$duration"
    else
        RECOVERY_STATUS="completed"
        log "INFO" "Emergency recovery completed successfully in ${duration} seconds"
        send_alert "resolved" "Emergency recovery completed successfully"
        
        # Generate success report
        generate_recovery_report "success" "$duration"
    fi
    
    # Clean up temporary files
    if [ -d "$RECOVERY_DIR" ] && [ "$RECOVERY_DIR" != "/" ]; then
        rm -rf "$RECOVERY_DIR" || true
    fi
    
    exit $exit_code
}

trap cleanup EXIT

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --backup-type)
            BACKUP_TYPE="$2"
            shift 2
            ;;
        --timestamp)
            TARGET_TIMESTAMP="$2"
            BACKUP_TYPE="point-in-time"
            shift 2
            ;;
        --backup-path)
            SPECIFIC_BACKUP="$2"
            BACKUP_TYPE="specific"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --skip-verification)
            SKIP_VERIFICATION=true
            shift
            ;;
        --force)
            FORCE_RECOVERY=true
            shift
            ;;
        --parallel-jobs)
            PARALLEL_JOBS="$2"
            shift 2
            ;;
        -h|--help)
            cat << EOF
Emergency Recovery Script - Restore Qdrant MCP from backup

Usage: $0 [options]

Options:
  --backup-type TYPE     Recovery type: latest, specific, point-in-time
  --timestamp TIMESTAMP  Restore to specific timestamp (YYYY-MM-DD HH:MM:SS)
  --backup-path PATH     Specific backup directory to restore from
  --dry-run              Show recovery plan without executing
  --skip-verification    Skip backup verification before recovery
  --force                Force recovery even if service appears healthy
  --parallel-jobs N      Number of parallel recovery jobs (default: 4)

Examples:
  $0                                    # Recover from latest backup
  $0 --timestamp "2024-01-15 14:30:00" # Point-in-time recovery
  $0 --backup-path /backups/full/20240115_143000 # Specific backup
  $0 --dry-run                         # Preview recovery plan
EOF
            exit 0
            ;;
        *)
            log "ERROR" "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

# Pre-recovery health check
pre_recovery_check() {
    log "INFO" "Performing pre-recovery health check..."
    
    # Check if Qdrant is running and accessible
    local qdrant_healthy=false
    if curl -s --max-time 5 "${QDRANT_URL}/healthz" > /dev/null; then
        qdrant_healthy=true
        log "INFO" "Qdrant service is currently accessible"
    else
        log "WARN" "Qdrant service is not accessible"
    fi
    
    # If service is healthy and force is not specified, confirm recovery
    if [ "$qdrant_healthy" = true ] && [ "$FORCE_RECOVERY" = false ] && [ "$DRY_RUN" = false ]; then
        log "WARN" "Qdrant service appears healthy. Recovery may cause data loss."
        echo "Qdrant service appears to be running. Continue with recovery? (y/N): "
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            log "INFO" "Recovery cancelled by user"
            exit 0
        fi
    fi
    
    # Check system resources
    local available_memory
    available_memory=$(free -m | awk 'NR==2{printf "%.0f", $7}')
    if [ "$available_memory" -lt 1024 ]; then
        log "WARN" "Low available memory: ${available_memory}MB"
    fi
    
    local available_space
    available_space=$(df /tmp | awk 'NR==2 {print $4}')
    if [ "$available_space" -lt 5242880 ]; then  # 5GB in KB
        log "ERROR" "Insufficient disk space for recovery"
        return 1
    fi
    
    log "INFO" "Pre-recovery health check completed"
}

# Find backup to restore from
find_backup_to_restore() {
    log "INFO" "Finding backup to restore (type: $BACKUP_TYPE)..."
    
    local backup_path=""
    
    case $BACKUP_TYPE in
        "latest")
            backup_path=$(find "$BACKUP_BASE_DIR" -name "manifest.json" -type f \
                | xargs ls -t | head -1 | xargs dirname)
            ;;
        "specific")
            backup_path="$SPECIFIC_BACKUP"
            ;;
        "point-in-time")
            if [ -z "$TARGET_TIMESTAMP" ]; then
                log "ERROR" "Target timestamp required for point-in-time recovery"
                return 1
            fi
            backup_path=$(find_backup_by_timestamp "$TARGET_TIMESTAMP")
            ;;
    esac
    
    if [ -z "$backup_path" ] || [ ! -d "$backup_path" ]; then
        log "ERROR" "Could not find valid backup to restore from"
        return 1
    fi
    
    if [ ! -f "$backup_path/manifest.json" ]; then
        log "ERROR" "Backup manifest not found: $backup_path/manifest.json"
        return 1
    fi
    
    log "INFO" "Selected backup for recovery: $backup_path"
    echo "$backup_path"
}

# Find backup by timestamp
find_backup_by_timestamp() {
    local target_timestamp=$1
    local target_epoch
    target_epoch=$(date -d "$target_timestamp" +%s 2>/dev/null || {
        log "ERROR" "Invalid timestamp format: $target_timestamp"
        return 1
    })
    
    local closest_backup=""
    local min_diff=999999999
    
    find "$BACKUP_BASE_DIR" -name "manifest.json" -type f | while read manifest; do
        local backup_dir
        backup_dir=$(dirname "$manifest")
        local backup_timestamp
        backup_timestamp=$(jq -r '.backup_timestamp // 0' "$manifest")
        
        if [ "$backup_timestamp" -gt 0 ]; then
            local diff=$((target_epoch - backup_timestamp))
            if [ $diff -ge 0 ] && [ $diff -lt $min_diff ]; then
                echo "$backup_dir"
                return
            fi
        fi
    done
}

# Verify backup integrity
verify_backup() {
    local backup_path=$1
    
    if [ "$SKIP_VERIFICATION" = true ]; then
        log "INFO" "Skipping backup verification (disabled)"
        return 0
    fi
    
    log "INFO" "Verifying backup integrity: $backup_path"
    
    local manifest="$backup_path/manifest.json"
    if [ ! -f "$manifest" ]; then
        log "ERROR" "Backup manifest not found"
        return 1
    fi
    
    # Verify manifest format
    if ! jq empty "$manifest" 2>/dev/null; then
        log "ERROR" "Invalid backup manifest format"
        return 1
    fi
    
    # Check if this is a Qdrant backup
    local backup_type
    backup_type=$(jq -r '.backup_type // "unknown"' "$manifest")
    
    # Verify collection snapshots
    local collections_count
    collections_count=$(jq -r '.collections_backed_up // .collections_count // 0' "$manifest")
    
    if [ "$collections_count" -gt 0 ]; then
        log "INFO" "Verifying $collections_count collection snapshots..."
        
        for meta_file in "$backup_path"/*.meta; do
            if [ ! -f "$meta_file" ]; then
                continue
            fi
            
            local collection_name
            collection_name=$(jq -r '.collection_name' "$meta_file" 2>/dev/null || basename "$meta_file" .meta)
            local snapshot_file="$backup_path/${collection_name}.snapshot"
            
            if [ ! -f "$snapshot_file" ]; then
                log "ERROR" "Snapshot file missing for collection: $collection_name"
                return 1
            fi
            
            # Verify checksum if available
            if jq -e '.checksum' "$meta_file" > /dev/null 2>&1; then
                local expected_checksum
                expected_checksum=$(jq -r '.checksum' "$meta_file")
                local actual_checksum
                actual_checksum=$(sha256sum "$snapshot_file" | cut -d' ' -f1)
                
                if [ "$expected_checksum" != "$actual_checksum" ]; then
                    log "ERROR" "Checksum mismatch for collection: $collection_name"
                    return 1
                fi
            fi
        done
    fi
    
    log "INFO" "Backup verification completed successfully"
}

# Stop Qdrant service safely
stop_qdrant_service() {
    log "INFO" "Stopping Qdrant service for recovery..."
    
    if [ "$DRY_RUN" = true ]; then
        log "INFO" "[DRY-RUN] Would stop Qdrant service"
        return 0
    fi
    
    # Try multiple methods to stop Qdrant
    local stopped=false
    
    # Method 1: systemd
    if command -v systemctl &> /dev/null; then
        if systemctl is-active --quiet qdrant 2>/dev/null; then
            log "INFO" "Stopping Qdrant via systemd..."
            systemctl stop qdrant && stopped=true
        fi
    fi
    
    # Method 2: Docker
    if [ "$stopped" = false ] && command -v docker &> /dev/null; then
        local qdrant_containers
        qdrant_containers=$(docker ps --format "{{.Names}}" | grep -i qdrant || true)
        if [ -n "$qdrant_containers" ]; then
            log "INFO" "Stopping Qdrant containers..."
            echo "$qdrant_containers" | xargs -r docker stop && stopped=true
        fi
    fi
    
    # Method 3: Kubernetes
    if [ "$stopped" = false ] && command -v kubectl &> /dev/null; then
        if kubectl get pods | grep -q qdrant; then
            log "INFO" "Scaling down Qdrant deployment..."
            kubectl scale deployment qdrant --replicas=0 && stopped=true
        fi
    fi
    
    # Method 4: Process termination
    if [ "$stopped" = false ]; then
        local qdrant_pids
        qdrant_pids=$(pgrep -f qdrant || true)
        if [ -n "$qdrant_pids" ]; then
            log "INFO" "Terminating Qdrant processes..."
            echo "$qdrant_pids" | xargs -r kill -TERM
            sleep 5
            # Force kill if still running
            qdrant_pids=$(pgrep -f qdrant || true)
            if [ -n "$qdrant_pids" ]; then
                echo "$qdrant_pids" | xargs -r kill -KILL
            fi
            stopped=true
        fi
    fi
    
    # Verify service is stopped
    sleep 2
    if curl -s --max-time 2 "${QDRANT_URL}/healthz" > /dev/null; then
        log "WARN" "Qdrant service may still be running"
    else
        log "INFO" "Qdrant service stopped successfully"
    fi
}

# Prepare recovery environment
prepare_recovery_environment() {
    log "INFO" "Preparing recovery environment..."
    
    if [ "$DRY_RUN" = true ]; then
        log "INFO" "[DRY-RUN] Would prepare recovery environment"
        return 0
    fi
    
    # Create recovery working directory
    mkdir -p "$RECOVERY_DIR"
    chmod 750 "$RECOVERY_DIR"
    
    # Clean existing Qdrant data directory (if safe to do so)
    local qdrant_data_dir="${QDRANT_DATA_DIR:-/var/lib/qdrant}"
    if [ -d "$qdrant_data_dir" ]; then
        log "INFO" "Backing up existing Qdrant data to recovery directory..."
        cp -r "$qdrant_data_dir" "$RECOVERY_DIR/qdrant-data-backup" || true
        
        log "INFO" "Clearing Qdrant data directory..."
        rm -rf "$qdrant_data_dir"/*
    fi
    
    log "INFO" "Recovery environment prepared"
}

# Restore collection from snapshot
restore_collection() {
    local collection=$1
    local snapshot_file=$2
    local meta_file=$3
    
    log "INFO" "Restoring collection: $collection"
    
    if [ "$DRY_RUN" = true ]; then
        log "INFO" "[DRY-RUN] Would restore collection: $collection"
        return 0
    fi
    
    local start_time
    start_time=$(date +%s)
    
    # Read collection metadata
    local collection_config
    if [ -f "$meta_file" ] && jq -e '.collection_config' "$meta_file" > /dev/null; then
        collection_config=$(jq '.collection_config' "$meta_file")
    else
        log "WARN" "Collection config not found in metadata for $collection"
        collection_config="{}"
    fi
    
    # Create collection if config is available
    if [ "$collection_config" != "{}" ]; then
        log "INFO" "Creating collection $collection with restored config..."
        
        local response
        response=$(curl -s -X PUT "${QDRANT_URL}/collections/${collection}" \
            -H "Content-Type: application/json" \
            -d "$collection_config")
        
        if echo "$response" | jq -e '.status == "ok"' > /dev/null; then
            log "INFO" "Collection $collection created successfully"
        else
            log "ERROR" "Failed to create collection $collection: $response"
            FAILED_COLLECTIONS+=("$collection")
            return 1
        fi
    fi
    
    # Upload snapshot
    log "INFO" "Uploading snapshot for collection: $collection"
    
    local upload_response
    upload_response=$(curl -s -X POST "${QDRANT_URL}/collections/${collection}/snapshots/upload" \
        -F "snapshot=@${snapshot_file}")
    
    if ! echo "$upload_response" | jq -e '.status == "ok"' > /dev/null; then
        log "ERROR" "Failed to upload snapshot for collection $collection: $upload_response"
        FAILED_COLLECTIONS+=("$collection")
        return 1
    fi
    
    # Get uploaded snapshot name
    local snapshot_name
    snapshot_name=$(echo "$upload_response" | jq -r '.result.name // "uploaded"')
    
    # Recover from snapshot
    log "INFO" "Recovering collection $collection from snapshot..."
    
    local recover_response
    recover_response=$(curl -s -X PUT "${QDRANT_URL}/collections/${collection}/snapshots/${snapshot_name}/recover")
    
    if echo "$recover_response" | jq -e '.status == "ok"' > /dev/null; then
        local end_time
        end_time=$(date +%s)
        local duration=$((end_time - start_time))
        
        log "INFO" "✓ Collection $collection restored successfully in ${duration}s"
        RECOVERED_COLLECTIONS+=("$collection")
        
        # Clean up uploaded snapshot
        curl -s -X DELETE "${QDRANT_URL}/collections/${collection}/snapshots/${snapshot_name}" > /dev/null || true
        
        return 0
    else
        log "ERROR" "Failed to recover collection $collection: $recover_response"
        FAILED_COLLECTIONS+=("$collection")
        return 1
    fi
}

# Start Qdrant service
start_qdrant_service() {
    log "INFO" "Starting Qdrant service..."
    
    if [ "$DRY_RUN" = true ]; then
        log "INFO" "[DRY-RUN] Would start Qdrant service"
        return 0
    fi
    
    # Try multiple methods to start Qdrant
    local started=false
    
    # Method 1: systemd
    if command -v systemctl &> /dev/null; then
        if systemctl list-unit-files | grep -q "^qdrant.service"; then
            log "INFO" "Starting Qdrant via systemd..."
            systemctl start qdrant && started=true
        fi
    fi
    
    # Method 2: Docker
    if [ "$started" = false ] && command -v docker &> /dev/null; then
        if docker ps -a --format "{{.Names}}" | grep -q qdrant; then
            log "INFO" "Starting Qdrant containers..."
            docker ps -a --format "{{.Names}}" | grep qdrant | xargs -r docker start && started=true
        fi
    fi
    
    # Method 3: Kubernetes
    if [ "$started" = false ] && command -v kubectl &> /dev/null; then
        if kubectl get deployment qdrant &> /dev/null; then
            log "INFO" "Scaling up Qdrant deployment..."
            kubectl scale deployment qdrant --replicas=3 && started=true
        fi
    fi
    
    if [ "$started" = false ]; then
        log "ERROR" "Could not start Qdrant service using any available method"
        return 1
    fi
    
    # Wait for service to be ready
    log "INFO" "Waiting for Qdrant service to be ready..."
    local retry_count=0
    local max_retries=30
    
    while [ $retry_count -lt $max_retries ]; do
        if curl -s --max-time 2 "${QDRANT_URL}/healthz" > /dev/null; then
            log "INFO" "Qdrant service is ready"
            return 0
        fi
        
        retry_count=$((retry_count + 1))
        sleep 2
    done
    
    log "ERROR" "Qdrant service failed to start within timeout"
    return 1
}

# Perform collection recovery
perform_collection_recovery() {
    local backup_path=$1
    
    log "INFO" "Starting collection recovery from: $backup_path"
    
    # Find all collection snapshots
    local collection_snapshots=()
    for snapshot_file in "$backup_path"/*.snapshot; do
        if [ -f "$snapshot_file" ]; then
            collection_snapshots+=("$snapshot_file")
        fi
    done
    
    if [ ${#collection_snapshots[@]} -eq 0 ]; then
        log "WARN" "No collection snapshots found in backup"
        return 0
    fi
    
    log "INFO" "Found ${#collection_snapshots[@]} collections to restore"
    
    # Restore collections in parallel
    restore_collection_wrapper() {
        local snapshot_file=$1
        local collection
        collection=$(basename "$snapshot_file" .snapshot)
        local meta_file="$backup_path/${collection}.meta"
        
        restore_collection "$collection" "$snapshot_file" "$meta_file"
    }
    
    export -f restore_collection restore_collection_wrapper log
    export QDRANT_URL DRY_RUN FAILED_COLLECTIONS RECOVERED_COLLECTIONS
    
    printf '%s\n' "${collection_snapshots[@]}" | \
        xargs -n 1 -P "$PARALLEL_JOBS" -I {} bash -c 'restore_collection_wrapper "$@"' _ {}
    
    log "INFO" "Collection recovery phase completed"
}

# Post-recovery verification
post_recovery_verification() {
    log "INFO" "Performing post-recovery verification..."
    
    if [ "$DRY_RUN" = true ]; then
        log "INFO" "[DRY-RUN] Would perform post-recovery verification"
        return 0
    fi
    
    # Check service health
    if ! curl -s --max-time 10 "${QDRANT_URL}/healthz" > /dev/null; then
        log "ERROR" "Qdrant service health check failed"
        return 1
    fi
    
    # Verify collections are accessible
    local collections_response
    collections_response=$(curl -s "${QDRANT_URL}/collections")
    
    if ! echo "$collections_response" | jq -e '.result' > /dev/null; then
        log "ERROR" "Could not retrieve collections list"
        return 1
    fi
    
    local recovered_count=${#RECOVERED_COLLECTIONS[@]}
    local actual_count
    actual_count=$(echo "$collections_response" | jq '.result.collections | length')
    
    log "INFO" "Collections expected: $recovered_count, actual: $actual_count"
    
    # Test basic functionality
    for collection in "${RECOVERED_COLLECTIONS[@]}"; do
        log "INFO" "Verifying collection: $collection"
        
        local collection_info
        collection_info=$(curl -s "${QDRANT_URL}/collections/${collection}")
        
        if echo "$collection_info" | jq -e '.result.status' > /dev/null; then
            local status
            status=$(echo "$collection_info" | jq -r '.result.status')
            local points_count
            points_count=$(echo "$collection_info" | jq -r '.result.points_count // 0')
            
            log "INFO" "Collection $collection: status=$status, points=$points_count"
        else
            log "ERROR" "Collection $collection verification failed"
            return 1
        fi
    done
    
    log "INFO" "Post-recovery verification completed successfully"
}

# Generate recovery report
generate_recovery_report() {
    local status=$1
    local duration=$2
    
    local report_file="/var/log/recovery/emergency-recovery-${RECOVERY_DATE}.json"
    mkdir -p "$(dirname "$report_file")"
    
    cat > "$report_file" << EOF
{
  "recovery_id": "emergency-recovery-${RECOVERY_DATE}",
  "status": "${status}",
  "start_time": ${START_TIME},
  "end_time": $(date +%s),
  "duration_seconds": ${duration},
  "backup_type": "${BACKUP_TYPE}",
  "collections_recovered": $(printf '%s\n' "${RECOVERED_COLLECTIONS[@]}" | jq -R . | jq -s .),
  "collections_failed": $(printf '%s\n' "${FAILED_COLLECTIONS[@]}" | jq -R . | jq -s .),
  "total_collections": $((${#RECOVERED_COLLECTIONS[@]} + ${#FAILED_COLLECTIONS[@]})),
  "success_rate": $(echo "scale=2; ${#RECOVERED_COLLECTIONS[@]} * 100 / ($((${#RECOVERED_COLLECTIONS[@]} + ${#FAILED_COLLECTIONS[@]})) || 1)" | bc),
  "hostname": "$(hostname)",
  "recovery_operator": "${USER:-system}"
}
EOF
    
    log "INFO" "Recovery report generated: $report_file"
}

# Main recovery function
main() {
    log "INFO" "=== EMERGENCY RECOVERY STARTED ==="
    log "INFO" "Recovery ID: emergency-recovery-${RECOVERY_DATE}"
    log "INFO" "Backup type: $BACKUP_TYPE"
    log "INFO" "Dry run: $DRY_RUN"
    
    send_alert "warning" "Emergency recovery started (dry-run: $DRY_RUN)"
    
    # Setup
    mkdir -p "$(dirname "$LOG_FILE")"
    
    # Pre-recovery checks
    pre_recovery_check
    
    # Find backup to restore from
    local backup_path
    backup_path=$(find_backup_to_restore)
    
    # Verify backup integrity
    verify_backup "$backup_path"
    
    if [ "$DRY_RUN" = true ]; then
        log "INFO" "=== DRY RUN SUMMARY ==="
        log "INFO" "Would restore from backup: $backup_path"
        
        # Show what would be recovered
        local collection_count=0
        for snapshot_file in "$backup_path"/*.snapshot; do
            if [ -f "$snapshot_file" ]; then
                local collection
                collection=$(basename "$snapshot_file" .snapshot)
                log "INFO" "Would restore collection: $collection"
                collection_count=$((collection_count + 1))
            fi
        done
        
        log "INFO" "Total collections to restore: $collection_count"
        log "INFO" "=== DRY RUN COMPLETED ==="
        return 0
    fi
    
    # Perform recovery
    stop_qdrant_service
    prepare_recovery_environment
    start_qdrant_service
    perform_collection_recovery "$backup_path"
    post_recovery_verification
    
    # Summary
    log "INFO" "=== RECOVERY SUMMARY ==="
    log "INFO" "Collections recovered: ${#RECOVERED_COLLECTIONS[@]}"
    log "INFO" "Collections failed: ${#FAILED_COLLECTIONS[@]}"
    
    if [ ${#FAILED_COLLECTIONS[@]} -eq 0 ]; then
        log "INFO" "✓ All collections recovered successfully"
        send_alert "resolved" "Emergency recovery completed successfully - all services restored"
    else
        log "WARN" "⚠ Some collections failed to recover: ${FAILED_COLLECTIONS[*]}"
        send_alert "warning" "Emergency recovery completed with partial failures"
    fi
    
    log "INFO" "=== EMERGENCY RECOVERY COMPLETED ==="
}

# Run main function
main "$@"