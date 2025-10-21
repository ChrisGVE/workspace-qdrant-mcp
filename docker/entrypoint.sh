#!/bin/bash
set -euo pipefail

# Entrypoint script for Workspace Qdrant MCP Docker container
# Handles initialization, configuration validation, and graceful startup

# Environment defaults
export WORKSPACE_QDRANT_HOST="${WORKSPACE_QDRANT_HOST:-0.0.0.0}"
export WORKSPACE_QDRANT_PORT="${WORKSPACE_QDRANT_PORT:-8000}"
export WORKSPACE_QDRANT_LOG_LEVEL="${WORKSPACE_QDRANT_LOG_LEVEL:-INFO}"
export WORKSPACE_QDRANT_DATA_DIR="${WORKSPACE_QDRANT_DATA_DIR:-/app/data}"
export WORKSPACE_QDRANT_LOG_DIR="${WORKSPACE_QDRANT_LOG_DIR:-/app/logs}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" >&2
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1" >&2
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" >&2
}

# Signal handlers for graceful shutdown
shutdown() {
    log_info "Received shutdown signal..."
    if [[ -n "${APP_PID:-}" ]]; then
        log_info "Stopping application (PID: $APP_PID)..."
        kill -TERM "$APP_PID" 2>/dev/null || true
        wait "$APP_PID" 2>/dev/null || true
    fi
    log_success "Application stopped gracefully"
    exit 0
}

trap shutdown SIGTERM SIGINT

# Health check function
health_check() {
    local max_attempts=30
    local attempt=1
    
    log_info "Performing health check..."
    
    while [[ $attempt -le $max_attempts ]]; do
        if python -c "
import sys
sys.path.insert(0, '/app/src')
try:
    from workspace_qdrant_mcp.core.client import QdrantClient
    client = QdrantClient()
    # Basic connectivity test
    log_success('Health check passed')
    sys.exit(0)
except Exception as e:
    print(f'Health check failed: {e}')
    sys.exit(1)
" 2>/dev/null; then
            log_success "Health check passed on attempt $attempt"
            return 0
        fi
        
        log_warn "Health check failed, attempt $attempt/$max_attempts"
        sleep 2
        ((attempt++))
    done
    
    log_error "Health check failed after $max_attempts attempts"
    return 1
}

# Validate configuration
validate_config() {
    log_info "Validating configuration..."
    
    # Check required directories exist and are writable
    for dir in "$WORKSPACE_QDRANT_DATA_DIR" "$WORKSPACE_QDRANT_LOG_DIR"; do
        if [[ ! -d "$dir" ]]; then
            log_error "Directory does not exist: $dir"
            return 1
        fi
        
        if [[ ! -w "$dir" ]]; then
            log_error "Directory is not writable: $dir"
            return 1
        fi
    done
    
    # Validate port
    if [[ ! "$WORKSPACE_QDRANT_PORT" =~ ^[0-9]+$ ]] || [[ "$WORKSPACE_QDRANT_PORT" -lt 1 ]] || [[ "$WORKSPACE_QDRANT_PORT" -gt 65535 ]]; then
        log_error "Invalid port: $WORKSPACE_QDRANT_PORT"
        return 1
    fi
    
    # Validate log level
    if [[ ! "$WORKSPACE_QDRANT_LOG_LEVEL" =~ ^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$ ]]; then
        log_error "Invalid log level: $WORKSPACE_QDRANT_LOG_LEVEL"
        return 1
    fi
    
    log_success "Configuration validation passed"
    return 0
}

# Initialize application
initialize() {
    log_info "Initializing Workspace Qdrant MCP..."
    
    # Ensure Python path is correct
    export PYTHONPATH="/app/src:${PYTHONPATH:-}"
    
    # Create log file with proper permissions
    local log_file="$WORKSPACE_QDRANT_LOG_DIR/workspace-qdrant-mcp.log"
    touch "$log_file" || {
        log_error "Cannot create log file: $log_file"
        return 1
    }
    
    # Test Python module import
    if ! python -c "
import sys
sys.path.insert(0, '/app/src')
import workspace_qdrant_mcp
print(f'Successfully imported workspace_qdrant_mcp version {workspace_qdrant_mcp.__version__}')
" 2>/dev/null; then
        log_error "Failed to import workspace_qdrant_mcp module"
        return 1
    fi
    
    log_success "Initialization complete"
}

# Start application based on command
start_application() {
    local cmd="$1"
    shift
    
    case "$cmd" in
        "workspace-qdrant-mcp"|"server")
            log_info "Starting MCP server on $WORKSPACE_QDRANT_HOST:$WORKSPACE_QDRANT_PORT"
            exec python -m workspace_qdrant_mcp.server "$@" &
            APP_PID=$!
            ;;
        "wqm")
            log_info "Starting WQM CLI: $*"
            exec python -m workspace_qdrant_mcp.cli.main "$@" &
            APP_PID=$!
            ;;
        "admin")
            log_info "Starting admin CLI: $*"
            exec python -m workspace_qdrant_mcp.utils.admin_cli "$@" &
            APP_PID=$!
            ;;
        "ingest")
            log_info "Starting document ingestion: $*"
            exec python -m workspace_qdrant_mcp.cli.ingest "$@" &
            APP_PID=$!
            ;;
        "health")
            log_info "Running health check: $*"
            exec python -m workspace_qdrant_mcp.cli.health "$@" &
            APP_PID=$!
            ;;
        "test"|"diagnostics")
            log_info "Running diagnostics: $*"
            exec python -m workspace_qdrant_mcp.cli.diagnostics "$@" &
            APP_PID=$!
            ;;
        "python")
            log_info "Starting Python interpreter: $*"
            exec python "$@" &
            APP_PID=$!
            ;;
        "bash"|"sh")
            log_info "Starting shell: $*"
            exec bash "$@" &
            APP_PID=$!
            ;;
        *)
            log_info "Starting custom command: $cmd $*"
            exec "$cmd" "$@" &
            APP_PID=$!
            ;;
    esac
    
    # Wait for application to start and perform health check for server mode
    if [[ "$cmd" == "workspace-qdrant-mcp" || "$cmd" == "server" ]]; then
        sleep 5  # Give server time to start
        if ! health_check; then
            log_error "Application failed health check during startup"
            return 1
        fi
    fi
    
    log_success "Application started successfully (PID: $APP_PID)"
}

# Main execution
main() {
    log_info "Starting Workspace Qdrant MCP container..."
    log_info "Version: 0.3.0"
    log_info "Host: $WORKSPACE_QDRANT_HOST"
    log_info "Port: $WORKSPACE_QDRANT_PORT"
    log_info "Log Level: $WORKSPACE_QDRANT_LOG_LEVEL"
    
    # Validate configuration
    if ! validate_config; then
        log_error "Configuration validation failed"
        exit 1
    fi
    
    # Initialize application
    if ! initialize; then
        log_error "Application initialization failed"
        exit 1
    fi
    
    # Handle special cases
    if [[ $# -eq 0 ]]; then
        set -- "workspace-qdrant-mcp"
    fi
    
    # Start application
    if ! start_application "$@"; then
        log_error "Failed to start application"
        exit 1
    fi
    
    # Wait for application
    if [[ -n "${APP_PID:-}" ]]; then
        wait "$APP_PID"
        local exit_code=$?
        if [[ $exit_code -ne 0 ]]; then
            log_error "Application exited with code $exit_code"
            exit $exit_code
        fi
    fi
    
    log_success "Container execution completed"
}

# Execute main function
main "$@"