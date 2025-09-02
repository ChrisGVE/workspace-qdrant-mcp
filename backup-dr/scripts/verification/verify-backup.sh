#!/bin/bash

# Backup Verification Script for Qdrant MCP Workspace
# Comprehensive verification of backup integrity and recoverability
# Usage: ./verify-backup.sh [options] [backup_path]

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${LOG_FILE:-/var/log/verification/backup-verification.log}"
TEMP_QDRANT_PORT=16333
TEMP_QDRANT_DIR="/tmp/qdrant-verify-$$"

# Default settings
BACKUP_PATH=""
VERIFICATION_TYPE="full"  # basic, full, restore-test
QUICK_CHECK=false
PARALLEL_JOBS=3
DRY_RUN=false

# Verification results
TOTAL_COLLECTIONS=0
VERIFIED_COLLECTIONS=0
FAILED_VERIFICATIONS=()
VERIFICATION_WARNINGS=()

# Logging function
log() {
    local level=$1
    shift
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$level] $*" | tee -a "$LOG_FILE"
}

# Error handling and cleanup
cleanup() {
    local exit_code=$?
    
    # Clean up temporary Qdrant instance
    if [ -d "$TEMP_QDRANT_DIR" ]; then
        log "INFO" "Cleaning up temporary verification environment..."
        
        # Stop temporary Qdrant if running
        if [ -f "$TEMP_QDRANT_DIR/qdrant.pid" ]; then
            local pid
            pid=$(cat "$TEMP_QDRANT_DIR/qdrant.pid")
            if kill -0 "$pid" 2>/dev/null; then
                kill "$pid" || true
                sleep 2
                kill -9 "$pid" 2>/dev/null || true
            fi
        fi
        
        rm -rf "$TEMP_QDRANT_DIR"
    fi
    
    if [ $exit_code -ne 0 ]; then
        log "ERROR" "Backup verification failed"
        generate_verification_report "failed"
    else
        log "INFO" "Backup verification completed"
        generate_verification_report "success"
    fi
    
    exit $exit_code
}

trap cleanup EXIT

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --type)
            VERIFICATION_TYPE="$2"
            shift 2
            ;;
        --quick)
            QUICK_CHECK=true
            shift
            ;;
        --parallel-jobs)
            PARALLEL_JOBS="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            cat << EOF
Backup Verification Script - Verify backup integrity and recoverability

Usage: $0 [options] [backup_path]

Options:
  --type TYPE         Verification type: basic, full, restore-test (default: full)
  --quick             Quick verification (skip detailed checks)
  --parallel-jobs N   Number of parallel verification jobs (default: 3)
  --dry-run           Show verification plan without executing

Verification Types:
  basic       - Basic file and checksum verification
  full        - Comprehensive verification including format validation
  restore-test - Full restoration test with temporary Qdrant instance

Examples:
  $0                                      # Verify latest backup
  $0 /backups/full/20240115_143000        # Verify specific backup
  $0 --type restore-test --quick          # Quick restore test
EOF
            exit 0
            ;;
        -*)
            log "ERROR" "Unknown parameter: $1"
            exit 1
            ;;
        *)
            BACKUP_PATH="$1"
            shift
            ;;
    esac
done

# Find backup to verify
find_backup_to_verify() {
    if [ -n "$BACKUP_PATH" ]; then
        if [ ! -d "$BACKUP_PATH" ]; then
            log "ERROR" "Backup path does not exist: $BACKUP_PATH"
            return 1
        fi
        echo "$BACKUP_PATH"
        return 0
    fi
    
    # Find latest backup
    local latest_backup
    latest_backup=$(find /backups -name "manifest.json" -type f \
        | xargs ls -t | head -1 | xargs dirname 2>/dev/null || echo "")
    
    if [ -z "$latest_backup" ]; then
        log "ERROR" "No backups found to verify"
        return 1
    fi
    
    log "INFO" "Using latest backup: $latest_backup"
    echo "$latest_backup"
}

# Basic backup verification
verify_backup_structure() {
    local backup_path=$1
    log "INFO" "Verifying backup structure: $backup_path"
    
    # Check manifest file
    if [ ! -f "$backup_path/manifest.json" ]; then
        log "ERROR" "Backup manifest not found"
        return 1
    fi
    
    # Validate manifest JSON
    if ! jq empty "$backup_path/manifest.json" 2>/dev/null; then
        log "ERROR" "Invalid backup manifest JSON format"
        return 1
    fi
    
    # Extract backup information
    local backup_type backup_date collections_count
    backup_type=$(jq -r '.backup_type // "unknown"' "$backup_path/manifest.json")
    backup_date=$(jq -r '.backup_date // "unknown"' "$backup_path/manifest.json")
    collections_count=$(jq -r '.collections_backed_up // .collections_count // 0' "$backup_path/manifest.json")
    
    log "INFO" "Backup type: $backup_type, Date: $backup_date, Collections: $collections_count"
    
    # Verify collection files exist
    if [ "$collections_count" -gt 0 ]; then
        local actual_snapshots
        actual_snapshots=$(find "$backup_path" -name "*.snapshot" | wc -l)
        
        if [ "$actual_snapshots" -ne "$collections_count" ]; then
            log "WARN" "Collection count mismatch: expected $collections_count, found $actual_snapshots"
            VERIFICATION_WARNINGS+=("Collection count mismatch")
        fi
    fi
    
    log "INFO" "Backup structure verification passed"
    TOTAL_COLLECTIONS=$collections_count
}

# Verify collection checksums
verify_collection_checksums() {
    local backup_path=$1
    log "INFO" "Verifying collection checksums..."
    
    local checksum_failures=0
    
    for meta_file in "$backup_path"/*.meta; do
        if [ ! -f "$meta_file" ]; then
            continue
        fi
        
        local collection_name
        collection_name=$(jq -r '.collection_name' "$meta_file" 2>/dev/null || basename "$meta_file" .meta)
        local snapshot_file="$backup_path/${collection_name}.snapshot"
        
        if [ ! -f "$snapshot_file" ]; then
            log "ERROR" "Snapshot file missing for collection: $collection_name"
            FAILED_VERIFICATIONS+=("$collection_name: missing snapshot")
            checksum_failures=$((checksum_failures + 1))
            continue
        fi
        
        # Verify file size
        if jq -e '.file_size' "$meta_file" > /dev/null 2>&1; then
            local expected_size actual_size
            expected_size=$(jq -r '.file_size' "$meta_file")
            actual_size=$(stat -f%z "$snapshot_file" 2>/dev/null || stat -c%s "$snapshot_file")
            
            if [ "$expected_size" -ne "$actual_size" ]; then
                log "ERROR" "File size mismatch for $collection_name: expected $expected_size, actual $actual_size"
                FAILED_VERIFICATIONS+=("$collection_name: size mismatch")
                checksum_failures=$((checksum_failures + 1))
                continue
            fi
        fi
        
        # Verify checksum
        if jq -e '.checksum' "$meta_file" > /dev/null 2>&1; then
            local expected_checksum actual_checksum
            expected_checksum=$(jq -r '.checksum' "$meta_file")
            actual_checksum=$(sha256sum "$snapshot_file" | cut -d' ' -f1)
            
            if [ "$expected_checksum" = "$actual_checksum" ]; then
                log "INFO" "✓ Checksum verified for collection: $collection_name"
                VERIFIED_COLLECTIONS=$((VERIFIED_COLLECTIONS + 1))
            else
                log "ERROR" "✗ Checksum mismatch for collection: $collection_name"
                FAILED_VERIFICATIONS+=("$collection_name: checksum mismatch")
                checksum_failures=$((checksum_failures + 1))
            fi
        else
            log "WARN" "No checksum available for collection: $collection_name"
            VERIFICATION_WARNINGS+=("$collection_name: no checksum")
        fi
    done
    
    if [ $checksum_failures -eq 0 ]; then
        log "INFO" "All collection checksums verified successfully"
        return 0
    else
        log "ERROR" "$checksum_failures collections failed checksum verification"
        return 1
    fi
}

# Verify snapshot file formats
verify_snapshot_formats() {
    local backup_path=$1
    
    if [ "$QUICK_CHECK" = true ]; then
        log "INFO" "Skipping format verification (quick mode)"
        return 0
    fi
    
    log "INFO" "Verifying snapshot file formats..."
    
    local format_failures=0
    
    for snapshot_file in "$backup_path"/*.snapshot; do
        if [ ! -f "$snapshot_file" ]; then
            continue
        fi
        
        local collection_name
        collection_name=$(basename "$snapshot_file" .snapshot)
        
        # Check if file is not empty
        if [ ! -s "$snapshot_file" ]; then
            log "ERROR" "Empty snapshot file for collection: $collection_name"
            FAILED_VERIFICATIONS+=("$collection_name: empty snapshot")
            format_failures=$((format_failures + 1))
            continue
        fi
        
        # Basic format validation (Qdrant snapshots are typically tar.gz)
        local file_type
        file_type=$(file "$snapshot_file" 2>/dev/null || echo "unknown")
        
        if echo "$file_type" | grep -q -E "(gzip|tar|archive)"; then
            log "INFO" "✓ Format OK for collection: $collection_name ($file_type)"
        else
            log "WARN" "Unexpected format for collection $collection_name: $file_type"
            VERIFICATION_WARNINGS+=("$collection_name: unexpected format")
        fi
        
        # Try to list archive contents (if it's a tar/gzip file)
        if echo "$file_type" | grep -q "gzip"; then
            if ! tar -tzf "$snapshot_file" > /dev/null 2>&1; then
                log "ERROR" "Corrupted archive for collection: $collection_name"
                FAILED_VERIFICATIONS+=("$collection_name: corrupted archive")
                format_failures=$((format_failures + 1))
            fi
        fi
    done
    
    if [ $format_failures -eq 0 ]; then
        log "INFO" "All snapshot formats verified successfully"
        return 0
    else
        log "ERROR" "$format_failures collections failed format verification"
        return 1
    fi
}

# Setup temporary Qdrant instance for restore testing
setup_temp_qdrant() {
    log "INFO" "Setting up temporary Qdrant instance for restore testing..."
    
    mkdir -p "$TEMP_QDRANT_DIR"
    
    # Check if Qdrant binary is available
    if ! command -v qdrant &> /dev/null; then
        log "WARN" "Qdrant binary not found, skipping restore test"
        return 1
    fi
    
    # Create temporary Qdrant configuration
    cat > "$TEMP_QDRANT_DIR/config.yaml" << EOF
service:
  host: 127.0.0.1
  port: $TEMP_QDRANT_PORT
  enable_cors: true

storage:
  storage_path: "$TEMP_QDRANT_DIR/storage"

cluster:
  enabled: false

log_level: WARN
EOF
    
    # Start temporary Qdrant instance
    log "INFO" "Starting temporary Qdrant instance on port $TEMP_QDRANT_PORT..."
    
    QDRANT_CONFIG_PATH="$TEMP_QDRANT_DIR/config.yaml" \
    nohup qdrant > "$TEMP_QDRANT_DIR/qdrant.log" 2>&1 &
    
    local qdrant_pid=$!
    echo "$qdrant_pid" > "$TEMP_QDRANT_DIR/qdrant.pid"
    
    # Wait for Qdrant to be ready
    local retry_count=0
    local max_retries=30
    
    while [ $retry_count -lt $max_retries ]; do
        if curl -s --max-time 2 "http://127.0.0.1:$TEMP_QDRANT_PORT/healthz" > /dev/null; then
            log "INFO" "Temporary Qdrant instance is ready"
            return 0
        fi
        
        retry_count=$((retry_count + 1))
        sleep 1
    done
    
    log "ERROR" "Failed to start temporary Qdrant instance"
    return 1
}

# Test restore of a single collection
test_restore_collection() {
    local collection=$1
    local snapshot_file=$2
    local meta_file=$3
    
    log "INFO" "Testing restore for collection: $collection"
    
    local temp_qdrant_url="http://127.0.0.1:$TEMP_QDRANT_PORT"
    
    # Read collection configuration if available
    local collection_config="{}"
    if [ -f "$meta_file" ] && jq -e '.collection_config' "$meta_file" > /dev/null; then
        collection_config=$(jq '.collection_config' "$meta_file")
    fi
    
    # Create collection
    if [ "$collection_config" != "{}" ]; then
        local create_response
        create_response=$(curl -s -X PUT "$temp_qdrant_url/collections/$collection" \
            -H "Content-Type: application/json" \
            -d "$collection_config")
        
        if ! echo "$create_response" | jq -e '.status == "ok"' > /dev/null; then
            log "ERROR" "Failed to create test collection $collection: $create_response"
            return 1
        fi
    fi
    
    # Upload snapshot
    local upload_response
    upload_response=$(curl -s -X POST "$temp_qdrant_url/collections/$collection/snapshots/upload" \
        -F "snapshot=@$snapshot_file")
    
    if ! echo "$upload_response" | jq -e '.status == "ok"' > /dev/null; then
        log "ERROR" "Failed to upload snapshot for $collection: $upload_response"
        return 1
    fi
    
    # Get uploaded snapshot name
    local snapshot_name
    snapshot_name=$(echo "$upload_response" | jq -r '.result.name // "uploaded"')
    
    # Restore from snapshot
    local restore_response
    restore_response=$(curl -s -X PUT "$temp_qdrant_url/collections/$collection/snapshots/$snapshot_name/recover")
    
    if ! echo "$restore_response" | jq -e '.status == "ok"' > /dev/null; then
        log "ERROR" "Failed to restore collection $collection: $restore_response"
        return 1
    fi
    
    # Verify collection after restore
    local collection_info
    collection_info=$(curl -s "$temp_qdrant_url/collections/$collection")
    
    if echo "$collection_info" | jq -e '.result' > /dev/null; then
        local points_count status
        points_count=$(echo "$collection_info" | jq -r '.result.points_count // 0')
        status=$(echo "$collection_info" | jq -r '.result.status // "unknown"')
        
        log "INFO" "✓ Collection $collection restored successfully: status=$status, points=$points_count"
        
        # Compare with expected values from metadata
        if [ -f "$meta_file" ] && jq -e '.collection_stats.points_count' "$meta_file" > /dev/null; then
            local expected_points
            expected_points=$(jq -r '.collection_stats.points_count' "$meta_file")
            
            if [ "$points_count" -ne "$expected_points" ]; then
                log "WARN" "Point count mismatch for $collection: expected $expected_points, got $points_count"
                VERIFICATION_WARNINGS+=("$collection: point count mismatch")
            fi
        fi
        
        # Clean up test collection
        curl -s -X DELETE "$temp_qdrant_url/collections/$collection" > /dev/null || true
        
        return 0
    else
        log "ERROR" "Collection $collection not accessible after restore"
        return 1
    fi
}

# Perform restore testing
perform_restore_test() {
    local backup_path=$1
    
    if [ "$VERIFICATION_TYPE" != "restore-test" ]; then
        log "INFO" "Skipping restore test (type: $VERIFICATION_TYPE)"
        return 0
    fi
    
    log "INFO" "Performing restore testing..."
    
    # Setup temporary Qdrant instance
    if ! setup_temp_qdrant; then
        log "WARN" "Could not setup temporary Qdrant, skipping restore test"
        return 0
    fi
    
    local restore_failures=0
    local collections_to_test=()
    
    # Find collections to test
    for snapshot_file in "$backup_path"/*.snapshot; do
        if [ -f "$snapshot_file" ]; then
            collections_to_test+=("$snapshot_file")
        fi
    done
    
    # Limit testing in quick mode
    if [ "$QUICK_CHECK" = true ] && [ ${#collections_to_test[@]} -gt 3 ]; then
        log "INFO" "Quick mode: testing only first 3 collections"
        collections_to_test=("${collections_to_test[@]:0:3}")
    fi
    
    log "INFO" "Testing restore of ${#collections_to_test[@]} collections..."
    
    # Test restores
    for snapshot_file in "${collections_to_test[@]}"; do
        local collection
        collection=$(basename "$snapshot_file" .snapshot)
        local meta_file="$backup_path/${collection}.meta"
        
        if test_restore_collection "$collection" "$snapshot_file" "$meta_file"; then
            VERIFIED_COLLECTIONS=$((VERIFIED_COLLECTIONS + 1))
        else
            FAILED_VERIFICATIONS+=("$collection: restore test failed")
            restore_failures=$((restore_failures + 1))
        fi
    done
    
    if [ $restore_failures -eq 0 ]; then
        log "INFO" "All restore tests passed successfully"
        return 0
    else
        log "ERROR" "$restore_failures collections failed restore testing"
        return 1
    fi
}

# Generate verification report
generate_verification_report() {
    local status=$1
    local report_file="/var/log/verification/backup-verification-$(date +%Y%m%d_%H%M%S).json"
    
    mkdir -p "$(dirname "$report_file")"
    
    cat > "$report_file" << EOF
{
  "verification_id": "backup-verification-$(date +%Y%m%d_%H%M%S)",
  "status": "$status",
  "verification_date": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "backup_path": "${BACKUP_PATH:-auto-detected}",
  "verification_type": "$VERIFICATION_TYPE",
  "quick_check": $QUICK_CHECK,
  "total_collections": $TOTAL_COLLECTIONS,
  "verified_collections": $VERIFIED_COLLECTIONS,
  "failed_verifications": $(printf '%s\n' "${FAILED_VERIFICATIONS[@]}" | jq -R . | jq -s . 2>/dev/null || echo '[]'),
  "warnings": $(printf '%s\n' "${VERIFICATION_WARNINGS[@]}" | jq -R . | jq -s . 2>/dev/null || echo '[]'),
  "success_rate": $(echo "scale=2; $VERIFIED_COLLECTIONS * 100 / ($TOTAL_COLLECTIONS || 1)" | bc 2>/dev/null || echo "0"),
  "hostname": "$(hostname)",
  "operator": "${USER:-system}"
}
EOF
    
    log "INFO" "Verification report generated: $report_file"
    
    # Export metrics for Prometheus
    local metrics_file="/var/lib/node_exporter/textfile_collector/backup_verification_metrics.prom"
    mkdir -p "$(dirname "$metrics_file")"
    
    local success_value=0
    if [ "$status" = "success" ]; then
        success_value=1
    fi
    
    cat > "$metrics_file" << EOF
# HELP backup_verification_success Indicates if the last backup verification was successful
# TYPE backup_verification_success gauge
backup_verification_success{type="$VERIFICATION_TYPE"} $success_value

# HELP backup_verification_collections_total Total number of collections verified
# TYPE backup_verification_collections_total gauge
backup_verification_collections_total{type="$VERIFICATION_TYPE"} $TOTAL_COLLECTIONS

# HELP backup_verification_collections_verified Number of collections that passed verification
# TYPE backup_verification_collections_verified gauge
backup_verification_collections_verified{type="$VERIFICATION_TYPE"} $VERIFIED_COLLECTIONS

# HELP backup_verification_timestamp_seconds Unix timestamp of the last verification
# TYPE backup_verification_timestamp_seconds gauge
backup_verification_timestamp_seconds{type="$VERIFICATION_TYPE"} $(date +%s)
EOF
}

# Main verification function
main() {
    log "INFO" "=== BACKUP VERIFICATION STARTED ==="
    log "INFO" "Verification type: $VERIFICATION_TYPE"
    log "INFO" "Quick check: $QUICK_CHECK"
    log "INFO" "Dry run: $DRY_RUN"
    
    # Setup
    mkdir -p "$(dirname "$LOG_FILE")"
    
    # Find backup to verify
    local backup_path
    backup_path=$(find_backup_to_verify)
    BACKUP_PATH="$backup_path"
    
    log "INFO" "Verifying backup: $backup_path"
    
    if [ "$DRY_RUN" = true ]; then
        log "INFO" "=== DRY RUN - VERIFICATION PLAN ==="
        log "INFO" "Would verify backup: $backup_path"
        log "INFO" "Verification type: $VERIFICATION_TYPE"
        
        # Show what would be verified
        if [ -f "$backup_path/manifest.json" ]; then
            local collections_count
            collections_count=$(jq -r '.collections_backed_up // .collections_count // 0' "$backup_path/manifest.json")
            log "INFO" "Collections to verify: $collections_count"
        fi
        
        log "INFO" "=== DRY RUN COMPLETED ==="
        return 0
    fi
    
    # Perform verification steps
    verify_backup_structure "$backup_path"
    
    case $VERIFICATION_TYPE in
        "basic")
            verify_collection_checksums "$backup_path"
            ;;
        "full")
            verify_collection_checksums "$backup_path"
            verify_snapshot_formats "$backup_path"
            ;;
        "restore-test")
            verify_collection_checksums "$backup_path"
            verify_snapshot_formats "$backup_path"
            perform_restore_test "$backup_path"
            ;;
    esac
    
    # Summary
    log "INFO" "=== VERIFICATION SUMMARY ==="
    log "INFO" "Total collections: $TOTAL_COLLECTIONS"
    log "INFO" "Verified collections: $VERIFIED_COLLECTIONS"
    log "INFO" "Failed verifications: ${#FAILED_VERIFICATIONS[@]}"
    log "INFO" "Warnings: ${#VERIFICATION_WARNINGS[@]}"
    
    if [ ${#FAILED_VERIFICATIONS[@]} -eq 0 ]; then
        log "INFO" "✓ All verifications passed successfully"
    else
        log "ERROR" "✗ Verification failures: ${FAILED_VERIFICATIONS[*]}"
        return 1
    fi
    
    if [ ${#VERIFICATION_WARNINGS[@]} -gt 0 ]; then
        log "WARN" "⚠ Verification warnings: ${VERIFICATION_WARNINGS[*]}"
    fi
    
    log "INFO" "=== BACKUP VERIFICATION COMPLETED ==="
}

# Run main function
main "$@"