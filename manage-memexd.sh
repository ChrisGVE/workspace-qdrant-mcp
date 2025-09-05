#!/bin/bash

# Workspace Qdrant MCP Daemon Management Script
# Provides cross-platform daemon management for memexd

set -euo pipefail

readonly DAEMON_NAME="memexd"
readonly DAEMON_PATH="/usr/local/bin/$DAEMON_NAME"
readonly CONFIG_DIR="$HOME/.workspace-qdrant-mcp"
readonly PID_FILE="$CONFIG_DIR/data/memexd.pid"
readonly LOG_FILE="$CONFIG_DIR/logs/memexd.log"
readonly LOCK_FILE="$CONFIG_DIR/data/memexd.lock"

# Colors
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m'

# Utility functions
info() { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Ensure directories exist
ensure_directories() {
    mkdir -p "$(dirname "$PID_FILE")"
    mkdir -p "$(dirname "$LOG_FILE")"
}

# Check if daemon binary exists
check_daemon_binary() {
    if [[ ! -x "$DAEMON_PATH" ]]; then
        error "Daemon binary not found at $DAEMON_PATH"
        error "Please run the installation script first"
        exit 1
    fi
}

# Get daemon PID if running
get_daemon_pid() {
    if [[ -f "$PID_FILE" ]]; then
        local pid
        pid=$(cat "$PID_FILE" 2>/dev/null || echo "")
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            echo "$pid"
            return 0
        else
            # Stale PID file
            rm -f "$PID_FILE"
        fi
    fi
    return 1
}

# Start daemon
start_daemon() {
    ensure_directories
    check_daemon_binary
    
    if pid=$(get_daemon_pid); then
        warning "Daemon is already running (PID: $pid)"
        return 0
    fi
    
    # Check for lock file (another process might be starting)
    if [[ -f "$LOCK_FILE" ]]; then
        local lock_pid
        lock_pid=$(cat "$LOCK_FILE" 2>/dev/null || echo "")
        if [[ -n "$lock_pid" ]] && kill -0 "$lock_pid" 2>/dev/null; then
            warning "Another process is already starting the daemon"
            return 1
        else
            rm -f "$LOCK_FILE"
        fi
    fi
    
    # Create lock file
    echo $$ > "$LOCK_FILE"
    
    info "Starting $DAEMON_NAME..."
    
    # Create log file with proper permissions
    touch "$LOG_FILE"
    
    # Start daemon with nohup and redirect output
    # Use exec to avoid shell wrapping
    nohup "$DAEMON_PATH" "$@" >"$LOG_FILE" 2>&1 </dev/null &
    local daemon_pid=$!
    
    # Save PID
    echo "$daemon_pid" > "$PID_FILE"
    
    # Remove lock file
    rm -f "$LOCK_FILE"
    
    # Wait a moment and verify startup
    sleep 2
    if kill -0 "$daemon_pid" 2>/dev/null; then
        success "Daemon started successfully (PID: $daemon_pid)"
        return 0
    else
        error "Daemon failed to start. Check logs: $LOG_FILE"
        rm -f "$PID_FILE"
        return 1
    fi
}

# Stop daemon
stop_daemon() {
    local pid
    if ! pid=$(get_daemon_pid); then
        warning "Daemon is not running"
        return 0
    fi
    
    info "Stopping $DAEMON_NAME (PID: $pid)..."
    
    # Send TERM signal first
    if kill -TERM "$pid" 2>/dev/null; then
        # Wait for graceful shutdown
        local count=0
        while kill -0 "$pid" 2>/dev/null && (( count < 30 )); do
            sleep 1
            ((count++))
        done
        
        # If still running, use KILL
        if kill -0 "$pid" 2>/dev/null; then
            warning "Daemon didn't stop gracefully, forcing termination..."
            kill -KILL "$pid" 2>/dev/null || true
            sleep 1
        fi
    fi
    
    # Clean up PID file
    rm -f "$PID_FILE"
    rm -f "$LOCK_FILE"
    
    # Verify termination
    if ! kill -0 "$pid" 2>/dev/null; then
        success "Daemon stopped successfully"
        return 0
    else
        error "Failed to stop daemon"
        return 1
    fi
}

# Restart daemon
restart_daemon() {
    stop_daemon
    sleep 2
    start_daemon "$@"
}

# Show daemon status
show_status() {
    local pid
    if pid=$(get_daemon_pid); then
        success "Daemon is running (PID: $pid)"
        
        # Show additional info if available
        if command -v ps >/dev/null 2>&1; then
            echo "Process info:"
            ps -p "$pid" -o pid,ppid,user,start,time,command 2>/dev/null || true
        fi
        
        # Show resource usage if available
        if [[ -f "/proc/$pid/status" ]]; then
            echo "Memory usage:"
            grep -E "VmRSS|VmSize" "/proc/$pid/status" 2>/dev/null || true
        elif command -v top >/dev/null 2>&1; then
            echo "Resource usage:"
            top -p "$pid" -l 1 2>/dev/null | grep "$pid" | head -1 || true
        fi
        
        return 0
    else
        warning "Daemon is not running"
        
        # Check if PID file exists but process is dead
        if [[ -f "$PID_FILE" ]]; then
            warning "Stale PID file found, cleaning up..."
            rm -f "$PID_FILE"
        fi
        
        return 1
    fi
}

# Show daemon logs
show_logs() {
    if [[ ! -f "$LOG_FILE" ]]; then
        warning "No log file found at $LOG_FILE"
        return 1
    fi
    
    local lines="${1:-50}"  # Default to last 50 lines
    
    if [[ "$lines" == "follow" ]] || [[ "$lines" == "-f" ]]; then
        info "Following daemon logs (Ctrl+C to stop)..."
        tail -f "$LOG_FILE"
    else
        info "Showing last $lines lines of daemon logs..."
        tail -n "$lines" "$LOG_FILE"
    fi
}

# Check daemon health
health_check() {
    local pid
    if ! pid=$(get_daemon_pid); then
        error "Daemon is not running"
        return 1
    fi
    
    info "Performing health check..."
    
    # Check if process is responsive (basic check)
    if kill -0 "$pid" 2>/dev/null; then
        success "Daemon process is responsive"
    else
        error "Daemon process is not responsive"
        return 1
    fi
    
    # Check log file for recent activity
    if [[ -f "$LOG_FILE" ]]; then
        local recent_lines
        recent_lines=$(tail -n 10 "$LOG_FILE" | wc -l)
        if (( recent_lines > 0 )); then
            info "Log file has recent activity"
        else
            warning "Log file appears empty or stale"
        fi
    else
        warning "No log file found"
    fi
    
    # Additional health checks could be added here
    # For example, checking if daemon is listening on expected ports
    
    success "Health check completed"
    return 0
}

# Show help
show_help() {
    cat << EOF
Workspace Qdrant MCP Daemon Management Script

USAGE:
    $0 <command> [options]

COMMANDS:
    start [args...]     Start the daemon (with optional arguments)
    stop               Stop the daemon
    restart [args...]  Restart the daemon (with optional arguments)
    status             Show daemon status
    logs [lines]       Show daemon logs (default: 50 lines)
                      Use 'follow' or '-f' to follow logs
    health             Perform health check
    help               Show this help message

EXAMPLES:
    $0 start                    # Start daemon with default settings
    $0 start --config /path     # Start daemon with custom config
    $0 stop                     # Stop daemon
    $0 restart                  # Restart daemon
    $0 status                   # Check if daemon is running
    $0 logs                     # Show last 50 log lines
    $0 logs 100                 # Show last 100 log lines
    $0 logs follow              # Follow logs in real-time
    $0 health                   # Check daemon health

FILES:
    Daemon binary: $DAEMON_PATH
    PID file:      $PID_FILE
    Log file:      $LOG_FILE
    Lock file:     $LOCK_FILE

For more information, see the project documentation.
EOF
}

# Main function
main() {
    case "${1:-help}" in
        start)
            shift
            start_daemon "$@"
            ;;
        stop)
            stop_daemon
            ;;
        restart)
            shift
            restart_daemon "$@"
            ;;
        status)
            show_status
            ;;
        logs)
            show_logs "${2:-50}"
            ;;
        health)
            health_check
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            error "Unknown command: $1"
            echo
            show_help
            exit 1
            ;;
    esac
}

# Run main function if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi