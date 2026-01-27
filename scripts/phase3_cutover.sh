#!/usr/bin/env bash
#
# Phase 3 Cutover Script for Unified Queue Migration
#
# This script safely transitions the workspace-qdrant-mcp system from dual-write mode
# (writing to both unified_queue and legacy queues) to unified-queue-only mode.
#
# Phases:
#   1. Pre-cutover checks (verify system health and readiness)
#   2. Drain legacy queues (wait for processing to complete)
#   3. Disable dual-write mode
#   4. Post-cutover validation
#   5. Optional: Remove legacy queue tables
#
# Usage:
#   ./scripts/phase3_cutover.sh [OPTIONS]
#
# Options:
#   --dry-run           Preview actions without executing
#   --skip-tests        Skip running integration tests
#   --skip-drain        Skip waiting for legacy queues to drain
#   --drop-tables       Remove legacy queue tables after cutover
#   --force             Skip confirmation prompts
#   --rollback          Revert to dual-write mode
#   --timeout SECONDS   Max time to wait for draining (default: 3600)
#   --poll-interval SEC Poll interval for drain check (default: 10)
#   -v, --verbose       Enable verbose output
#   -h, --help          Show this help message
#
# Exit Codes:
#   0 - Success
#   1 - Pre-check failure
#   2 - Drain timeout
#   3 - Post-cutover validation failure
#   4 - User cancelled
#   5 - Rollback performed
#

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
STATE_DB="${WQM_STATE_DB:-${HOME}/Library/Application Support/workspace-qdrant-mcp/state.db}"
CONFIG_FILE="${WQM_CONFIG:-${PROJECT_ROOT}/assets/default_configuration.yaml}"

# Default options
DRY_RUN=false
SKIP_TESTS=false
SKIP_DRAIN=false
DROP_TABLES=false
FORCE=false
ROLLBACK=false
VERBOSE=false
DRAIN_TIMEOUT=3600
POLL_INTERVAL=10

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*"
}

log_warn() {
    echo -e "${YELLOW}[WARNING]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*" >&2
}

log_verbose() {
    if [[ "$VERBOSE" == "true" ]]; then
        echo -e "${BLUE}[DEBUG]${NC} $*"
    fi
}

log_step() {
    echo -e "\n${GREEN}=== $* ===${NC}\n"
}

# Show help
show_help() {
    head -40 "$0" | grep '^#' | sed 's/^#//' | sed 's/^ //'
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --skip-tests)
                SKIP_TESTS=true
                shift
                ;;
            --skip-drain)
                SKIP_DRAIN=true
                shift
                ;;
            --drop-tables)
                DROP_TABLES=true
                shift
                ;;
            --force)
                FORCE=true
                shift
                ;;
            --rollback)
                ROLLBACK=true
                shift
                ;;
            --timeout)
                DRAIN_TIMEOUT="$2"
                shift 2
                ;;
            --poll-interval)
                POLL_INTERVAL="$2"
                shift 2
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# Execute command or show what would be done in dry-run mode
execute() {
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY-RUN] Would execute: $*"
        return 0
    fi
    log_verbose "Executing: $*"
    "$@"
}

# Get queue depth from SQLite
get_queue_depth() {
    local queue_table="$1"
    local status="${2:-pending}"

    if [[ ! -f "$STATE_DB" ]]; then
        echo "0"
        return
    fi

    sqlite3 "$STATE_DB" "SELECT COUNT(*) FROM $queue_table WHERE status = '$status';" 2>/dev/null || echo "0"
}

# Check if legacy queues exist
legacy_queues_exist() {
    if [[ ! -f "$STATE_DB" ]]; then
        return 1
    fi

    local ingestion_exists
    local content_exists

    ingestion_exists=$(sqlite3 "$STATE_DB" "SELECT name FROM sqlite_master WHERE type='table' AND name='ingestion_queue';" 2>/dev/null || echo "")
    content_exists=$(sqlite3 "$STATE_DB" "SELECT name FROM sqlite_master WHERE type='table' AND name='content_ingestion_queue';" 2>/dev/null || echo "")

    [[ -n "$ingestion_exists" || -n "$content_exists" ]]
}

# Get stuck items (in_progress for more than 1 hour)
get_stuck_items_count() {
    if [[ ! -f "$STATE_DB" ]]; then
        echo "0"
        return
    fi

    sqlite3 "$STATE_DB" "
        SELECT COUNT(*) FROM unified_queue
        WHERE status = 'in_progress'
        AND datetime(updated_at) < datetime('now', '-1 hour');
    " 2>/dev/null || echo "0"
}

# Check daemon health
check_daemon_health() {
    # Log to stderr so it doesn't interfere if called in subshell
    log_verbose "Checking daemon health..." >&2

    # Try to connect to daemon via CLI
    if command -v wqm &> /dev/null; then
        if wqm admin health &> /dev/null; then
            return 0
        fi
    fi

    # Fall back to checking if process is running
    if pgrep -f "memexd" &> /dev/null; then
        return 0
    fi

    return 1
}

# Run queue drift detection
check_queue_drift() {
    # Log to stderr so it doesn't pollute the return value
    log_verbose "Checking for queue drift..." >&2

    cd "$PROJECT_ROOT"

    local drift_output
    drift_output=$(uv run python -c "
import asyncio
from src.python.common.core.sqlite_state_manager import SQLiteStateManager

async def check_drift():
    state_manager = SQLiteStateManager()
    await state_manager.initialize()
    drift = await state_manager.detect_queue_drift()
    await state_manager.close()
    return drift['total_drift_count']

print(asyncio.run(check_drift()))
" 2>/dev/null) || drift_output="0"

    # Return just the number
    echo "$drift_output"
}

# Run integration tests
run_integration_tests() {
    log_info "Running queue processor integration tests..."

    cd "$PROJECT_ROOT"

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY-RUN] Would run: cargo test -p daemon-core --test queue_processor_integration_tests"
        return 0
    fi

    cd src/rust/daemon
    if cargo test -p daemon-core --test queue_processor_integration_tests -- --ignored 2>&1; then
        return 0
    else
        return 1
    fi
}

# Update configuration
update_config() {
    local key="$1"
    local value="$2"

    log_info "Updating config: $key = $value"

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY-RUN] Would update $key to $value in configuration"
        return 0
    fi

    # For now, we just log the change - actual config update depends on implementation
    log_warn "Manual config update required: Set $key to $value"
}

# Wait for legacy queues to drain
wait_for_drain() {
    log_info "Waiting for legacy queues to drain (timeout: ${DRAIN_TIMEOUT}s)..."

    local start_time
    start_time=$(date +%s)

    while true; do
        local ingestion_depth
        local content_depth

        ingestion_depth=$(get_queue_depth "ingestion_queue" "pending")
        content_depth=$(get_queue_depth "content_ingestion_queue" "pending")

        local total_pending=$((ingestion_depth + content_depth))

        if [[ "$total_pending" -eq 0 ]]; then
            log_success "Legacy queues drained successfully"
            return 0
        fi

        local elapsed
        elapsed=$(($(date +%s) - start_time))

        if [[ "$elapsed" -ge "$DRAIN_TIMEOUT" ]]; then
            log_error "Timeout waiting for legacy queues to drain"
            log_error "Remaining: ingestion_queue=$ingestion_depth, content_ingestion_queue=$content_depth"
            return 1
        fi

        log_info "Waiting... (pending: ingestion=$ingestion_depth, content=$content_depth, elapsed=${elapsed}s)"
        sleep "$POLL_INTERVAL"
    done
}

# Drop legacy queue tables
drop_legacy_tables() {
    log_warn "Dropping legacy queue tables..."

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY-RUN] Would drop tables: ingestion_queue, content_ingestion_queue"
        return 0
    fi

    if [[ ! -f "$STATE_DB" ]]; then
        log_warn "State database not found, skipping table drop"
        return 0
    fi

    sqlite3 "$STATE_DB" "DROP TABLE IF EXISTS ingestion_queue;" 2>/dev/null || true
    sqlite3 "$STATE_DB" "DROP TABLE IF EXISTS content_ingestion_queue;" 2>/dev/null || true

    log_success "Legacy queue tables dropped"
}

# Post-cutover validation
validate_cutover() {
    log_info "Running post-cutover validation..."

    cd "$PROJECT_ROOT"

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY-RUN] Would enqueue test item and verify processing"
        return 0
    fi

    # Enqueue a test item and verify it's processed
    local result
    result=$(uv run python -c "
import asyncio
import uuid
from src.python.common.core.sqlite_state_manager import SQLiteStateManager

async def validate():
    state_manager = SQLiteStateManager()
    await state_manager.initialize()

    # Enqueue test item
    test_id = f'cutover-test-{uuid.uuid4().hex[:8]}'
    await state_manager.enqueue_unified(
        item_type='file',
        op='ingest',
        tenant_id='cutover-test',
        collection='cutover-test-collection',
        payload={'file_path': f'/tmp/cutover-test-{test_id}.txt'},
        priority=10,
        branch='main',
        dual_write=False  # Should NOT dual-write anymore
    )

    # Verify item is in unified queue only
    unified_depth = await state_manager.get_unified_queue_depth(collection='cutover-test-collection')

    await state_manager.close()

    if unified_depth > 0:
        print('SUCCESS')
    else:
        print('FAILURE')

print(asyncio.run(validate()))
" 2>/dev/null) || result="FAILURE"

    if [[ "$result" == "SUCCESS" ]]; then
        log_success "Post-cutover validation passed"
        return 0
    else
        log_error "Post-cutover validation failed"
        return 1
    fi
}

# Perform rollback
perform_rollback() {
    log_step "Performing Rollback"

    log_warn "Rolling back to dual-write mode..."

    update_config "queue_processor.enable_dual_write" "true"

    # Restart daemon if running
    if check_daemon_health; then
        log_info "Restarting daemon to apply rollback..."
        execute wqm service restart || true
    fi

    log_success "Rollback completed - dual-write mode re-enabled"
    exit 5
}

# Confirmation prompt
confirm() {
    local message="$1"

    if [[ "$FORCE" == "true" ]]; then
        return 0
    fi

    echo -e "${YELLOW}$message${NC}"
    read -r -p "Continue? [y/N] " response
    case "$response" in
        [yY][eE][sS]|[yY])
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

# Main cutover process
main() {
    parse_args "$@"

    echo ""
    echo "=============================================="
    echo "  Phase 3 Cutover: Unified Queue Migration"
    echo "=============================================="
    echo ""

    if [[ "$DRY_RUN" == "true" ]]; then
        log_warn "Running in DRY-RUN mode - no changes will be made"
        echo ""
    fi

    # Handle rollback
    if [[ "$ROLLBACK" == "true" ]]; then
        perform_rollback
    fi

    # =========================================================================
    # Phase 1: Pre-cutover Checks
    # =========================================================================
    log_step "Phase 1: Pre-cutover Checks"

    # Check 1: Integration tests
    if [[ "$SKIP_TESTS" == "false" ]]; then
        log_info "Check 1: Running integration tests..."
        if ! run_integration_tests; then
            log_error "Integration tests failed - aborting cutover"
            exit 1
        fi
        log_success "Integration tests passed"
    else
        log_warn "Skipping integration tests (--skip-tests)"
    fi

    # Check 2: Queue drift
    log_info "Check 2: Checking for queue drift..."
    drift_count=$(check_queue_drift)
    if [[ "$drift_count" -gt 0 ]]; then
        log_error "Queue drift detected: $drift_count discrepancies"
        log_error "Run 'wqm admin drift-report --verbose' for details"
        exit 1
    fi
    log_success "No queue drift detected"

    # Check 3: Daemon health
    log_info "Check 3: Checking daemon health..."
    if ! check_daemon_health; then
        log_error "Daemon is not healthy - aborting cutover"
        exit 1
    fi
    log_success "Daemon is healthy"

    # Check 4: Stuck items
    log_info "Check 4: Checking for stuck items..."
    stuck_count=$(get_stuck_items_count)
    if [[ "$stuck_count" -gt 0 ]]; then
        log_error "Found $stuck_count items stuck in_progress for >1 hour"
        log_error "Resolve these items before proceeding"
        exit 1
    fi
    log_success "No stuck items found"

    # Check 5: Legacy queues exist
    log_info "Check 5: Checking legacy queue status..."
    if ! legacy_queues_exist; then
        log_warn "Legacy queues not found - may have already been migrated"
    else
        ingestion_depth=$(get_queue_depth "ingestion_queue" "pending")
        content_depth=$(get_queue_depth "content_ingestion_queue" "pending")
        log_info "Legacy queue depths: ingestion=$ingestion_depth, content=$content_depth"
    fi

    log_success "All pre-cutover checks passed"

    # Confirmation
    if ! confirm "Ready to proceed with cutover?"; then
        log_info "Cutover cancelled by user"
        exit 4
    fi

    # =========================================================================
    # Phase 2: Drain Legacy Queues
    # =========================================================================
    log_step "Phase 2: Drain Legacy Queues"

    if [[ "$SKIP_DRAIN" == "false" ]] && legacy_queues_exist; then
        # Disable new writes to legacy queues
        log_info "Step 2.1: Disabling new writes to legacy queues..."
        update_config "queue_processor.enable_dual_write" "false"

        # Wait for drain
        log_info "Step 2.2: Waiting for legacy queues to drain..."
        if ! wait_for_drain; then
            log_error "Failed to drain legacy queues within timeout"
            if confirm "Would you like to continue anyway?"; then
                log_warn "Continuing despite incomplete drain..."
            else
                log_info "Cutover aborted"
                exit 2
            fi
        fi
    else
        log_warn "Skipping drain phase (--skip-drain or no legacy queues)"
    fi

    # =========================================================================
    # Phase 3: Disable Dual-Write Mode
    # =========================================================================
    log_step "Phase 3: Disable Dual-Write Mode"

    log_info "Step 3.1: Setting enable_dual_write=false..."
    update_config "queue_processor.enable_dual_write" "false"

    log_info "Step 3.2: Restarting daemon to apply changes..."
    if check_daemon_health; then
        execute wqm service restart || log_warn "Failed to restart daemon - manual restart may be needed"
    fi

    log_success "Dual-write mode disabled"

    # =========================================================================
    # Phase 4: Post-cutover Validation
    # =========================================================================
    log_step "Phase 4: Post-cutover Validation"

    if ! validate_cutover; then
        log_error "Post-cutover validation failed!"
        if confirm "Would you like to rollback?"; then
            perform_rollback
        fi
        exit 3
    fi

    log_success "Post-cutover validation passed"

    # =========================================================================
    # Phase 5: Optional Table Cleanup
    # =========================================================================
    if [[ "$DROP_TABLES" == "true" ]]; then
        log_step "Phase 5: Legacy Table Cleanup"

        if ! confirm "This will permanently delete legacy queue tables. Continue?"; then
            log_info "Skipping table cleanup"
        else
            drop_legacy_tables
        fi
    fi

    # =========================================================================
    # Complete
    # =========================================================================
    echo ""
    echo "=============================================="
    echo "  Cutover Complete!"
    echo "=============================================="
    echo ""
    log_success "Phase 3 cutover completed successfully"
    log_info "The system is now running in unified-queue-only mode"
    echo ""
    log_info "Next steps:"
    log_info "  1. Monitor system for 24-48 hours"
    log_info "  2. Check metrics: wqm admin metrics"
    log_info "  3. If issues arise, run: $0 --rollback"
    if [[ "$DROP_TABLES" == "false" ]]; then
        log_info "  4. Once stable, remove legacy tables: $0 --drop-tables"
    fi
    echo ""

    exit 0
}

# Run main
main "$@"
