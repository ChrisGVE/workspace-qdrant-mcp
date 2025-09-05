#!/bin/bash
set -euo pipefail

# Workspace Qdrant MCP - Global Installation Script
# This script installs all three components of the system:
# 1. Rust daemon (memexd) - installed to /usr/local/bin/
# 2. Python MCP server - installed via uv tool install
# 3. Configuration and directory setup

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly INSTALL_LOG="/tmp/workspace-qdrant-mcp-install.log"
readonly CONFIG_DIR="$HOME/.workspace-qdrant-mcp"
readonly DAEMON_NAME="memexd"
readonly DAEMON_TARGET="/usr/local/bin/$DAEMON_NAME"

# Colors for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m' # No Color

# Logging function
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') $1" | tee -a "$INSTALL_LOG"
}

info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$INSTALL_LOG"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$INSTALL_LOG"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$INSTALL_LOG"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$INSTALL_LOG"
    exit 1
}

# Platform detection
detect_platform() {
    local os_name arch
    os_name="$(uname -s)"
    arch="$(uname -m)"
    
    case "$os_name" in
        Darwin)
            case "$arch" in
                x86_64) echo "macos-intel" ;;
                arm64) echo "macos-arm" ;;
                *) error "Unsupported macOS architecture: $arch" ;;
            esac
            ;;
        Linux)
            case "$arch" in
                x86_64) echo "linux-x64" ;;
                aarch64|arm64) echo "linux-arm64" ;;
                *) error "Unsupported Linux architecture: $arch" ;;
            esac
            ;;
        *)
            error "Unsupported operating system: $os_name"
            ;;
    esac
}

# Check prerequisites
check_prerequisites() {
    info "Checking prerequisites..."
    
    # Check for uv
    if ! command -v uv >/dev/null 2>&1; then
        error "uv is required but not installed. Install from: https://docs.astral.sh/uv/getting-started/installation/"
    fi
    
    # Check for Rust toolchain
    if ! command -v rustc >/dev/null 2>&1 || ! command -v cargo >/dev/null 2>&1; then
        error "Rust toolchain is required. Install from: https://rustup.rs/"
    fi
    
    # Check Python version via uv
    local python_version
    python_version=$(uv python list 2>/dev/null | grep -E 'python3\.[0-9]+' | head -1 | grep -oE '[0-9]+\.[0-9]+' || echo "")
    if [[ -z "$python_version" ]]; then
        # Fallback to system python
        python_version=$(python3 --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+' || echo "")
    fi
    if [[ -z "$python_version" ]]; then
        error "Could not determine Python version. Please ensure Python 3.10+ is installed."
    fi
    
    local major minor
    major=$(echo "$python_version" | cut -d. -f1)
    minor=$(echo "$python_version" | cut -d. -f2)
    
    if (( major < 3 || (major == 3 && minor < 10) )); then
        error "Python 3.10+ is required, found: $python_version"
    fi
    
    success "All prerequisites found (uv, rust, python $python_version)"
}

# Build and install daemon
install_daemon() {
    info "Building and installing Rust daemon..."
    
    # Change to rust-engine directory
    cd "$SCRIPT_DIR/rust-engine"
    
    # Build in release mode
    info "Building memexd in release mode..."
    cargo build --release --bin memexd
    
    # Verify binary exists
    local daemon_binary="$SCRIPT_DIR/rust-engine/target/release/$DAEMON_NAME"
    if [[ ! -f "$daemon_binary" ]]; then
        error "Failed to build daemon binary"
    fi
    
    # Install to /usr/local/bin (requires sudo)
    info "Installing daemon to $DAEMON_TARGET (requires sudo)..."
    if [[ -f "$DAEMON_TARGET" ]]; then
        warning "Existing daemon found at $DAEMON_TARGET, backing up..."
        sudo mv "$DAEMON_TARGET" "$DAEMON_TARGET.backup.$(date +%s)"
    fi
    
    sudo cp "$daemon_binary" "$DAEMON_TARGET"
    sudo chmod +x "$DAEMON_TARGET"
    
    # Verify installation
    if "$DAEMON_TARGET" --version >/dev/null 2>&1; then
        success "Daemon installed successfully"
    else
        error "Daemon installation verification failed"
    fi
    
    cd "$SCRIPT_DIR"
}

# Install Python package
install_server() {
    info "Installing Python MCP server..."
    
    # Check if already installed
    if uv tool list | grep -q "workspace-qdrant-mcp"; then
        warning "Package already installed, upgrading..."
        uv tool upgrade workspace-qdrant-mcp || uv tool install --force .
    else
        # Install from current directory
        uv tool install .
    fi
    
    # Verify installation
    if command -v workspace-qdrant-mcp >/dev/null 2>&1 && command -v wqm >/dev/null 2>&1; then
        success "MCP server installed successfully"
    else
        error "MCP server installation verification failed"
    fi
}

# Create directory structure
create_directories() {
    info "Creating configuration directories..."
    
    mkdir -p "$CONFIG_DIR"
    mkdir -p "$CONFIG_DIR/logs"
    mkdir -p "$CONFIG_DIR/data"
    mkdir -p "$CONFIG_DIR/config"
    
    success "Directory structure created at $CONFIG_DIR"
}

# Generate default configuration
generate_config() {
    info "Generating default configuration..."
    
    local config_file="$CONFIG_DIR/config/default.yaml"
    
    # Create default config based on example
    cat > "$config_file" << 'EOF'
# Default configuration for workspace-qdrant-mcp
# Generated by installation script

# Server configuration
host: "127.0.0.1"
port: 8000
debug: false

# Qdrant database configuration
qdrant:
  url: "http://localhost:6333"
  api_key: null  # Set to your API key for Qdrant Cloud
  timeout: 30
  prefer_grpc: false

# Embedding configuration
embedding:
  model: "sentence-transformers/all-MiniLM-L6-v2"
  enable_sparse_vectors: true
  chunk_size: 800
  chunk_overlap: 120
  batch_size: 50

# Workspace configuration
workspace:
  collection_suffixes: ["project", "docs"]  # Default collections
  global_collections: []  # Collections available across all projects
  github_user: null  # Set to your GitHub username for improved project detection
  collection_prefix: ""  # Optional prefix for all collection names
  max_collections: 100  # Maximum number of collections per workspace

# Auto-ingestion configuration
auto_ingestion:
  enabled: true  # Enable automatic file ingestion on server startup
  auto_create_watches: true  # Automatically create file watches for project directories
  include_common_files: true  # Include common document types (*.md, *.txt, *.pdf, etc.)
  include_source_files: true  # Include source code files (*.py, *.js, *.ts, etc.)
  target_collection_suffix: "project"  # Which collection suffix to use for auto-ingested files
  max_files_per_batch: 5  # Maximum files to process simultaneously
  batch_delay_seconds: 2.0  # Delay between processing batches
  max_file_size_mb: 50  # Maximum file size to process
  recursive_depth: 5  # Maximum directory depth to scan
  debounce_seconds: 10  # File change debounce time for watches

# Logging configuration
logging:
  level: "INFO"
  file: "~/.workspace-qdrant-mcp/logs/workspace-qdrant-mcp.log"
  max_size: "100MB"
  backup_count: 5
EOF
    
    success "Default configuration created at $config_file"
    
    # Set environment variable for easy access
    echo "export WORKSPACE_QDRANT_CONFIG=\"$config_file\"" >> "$HOME/.bashrc" 2>/dev/null || true
    echo "export WORKSPACE_QDRANT_CONFIG=\"$config_file\"" >> "$HOME/.zshrc" 2>/dev/null || true
}

# Create MCP configuration template
create_mcp_config() {
    info "Creating MCP configuration template..."
    
    local mcp_config="$CONFIG_DIR/config/mcp.json"
    
    cat > "$mcp_config" << EOF
{
  "mcpServers": {
    "workspace-qdrant": {
      "command": "workspace-qdrant-mcp",
      "args": ["--config-file", "$CONFIG_DIR/config/default.yaml"],
      "env": {
        "WORKSPACE_QDRANT_CONFIG": "$CONFIG_DIR/config/default.yaml"
      }
    }
  }
}
EOF
    
    success "MCP configuration template created at $mcp_config"
    info "To use with Claude, copy this file to your Claude configuration directory"
}

# Create daemon management script
create_daemon_manager() {
    info "Creating daemon management script..."
    
    local daemon_script="$CONFIG_DIR/manage-daemon.sh"
    
    cat > "$daemon_script" << 'EOF'
#!/bin/bash

# Workspace Qdrant MCP Daemon Manager

DAEMON_NAME="memexd"
DAEMON_PATH="/usr/local/bin/$DAEMON_NAME"
PID_FILE="$HOME/.workspace-qdrant-mcp/data/memexd.pid"
LOG_FILE="$HOME/.workspace-qdrant-mcp/logs/memexd.log"

case "$1" in
    start)
        if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
            echo "Daemon is already running (PID: $(cat "$PID_FILE"))"
            exit 1
        fi
        
        echo "Starting $DAEMON_NAME..."
        nohup "$DAEMON_PATH" > "$LOG_FILE" 2>&1 &
        echo $! > "$PID_FILE"
        echo "Daemon started (PID: $!)"
        ;;
    
    stop)
        if [[ -f "$PID_FILE" ]]; then
            local pid
            pid=$(cat "$PID_FILE")
            if kill -0 "$pid" 2>/dev/null; then
                kill "$pid"
                rm -f "$PID_FILE"
                echo "Daemon stopped"
            else
                echo "Daemon was not running"
                rm -f "$PID_FILE"
            fi
        else
            echo "No PID file found"
        fi
        ;;
    
    restart)
        "$0" stop
        sleep 2
        "$0" start
        ;;
    
    status)
        if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
            echo "Daemon is running (PID: $(cat "$PID_FILE"))"
        else
            echo "Daemon is not running"
            exit 1
        fi
        ;;
    
    logs)
        if [[ -f "$LOG_FILE" ]]; then
            tail -f "$LOG_FILE"
        else
            echo "No log file found at $LOG_FILE"
        fi
        ;;
    
    *)
        echo "Usage: $0 {start|stop|restart|status|logs}"
        exit 1
        ;;
esac
EOF
    
    chmod +x "$daemon_script"
    success "Daemon management script created at $daemon_script"
}

# Start daemon with default configuration
start_daemon() {
    info "Starting daemon with default configuration..."
    
    # Use the daemon manager script
    "$CONFIG_DIR/manage-daemon.sh" start
    
    # Wait a moment and verify
    sleep 3
    if "$CONFIG_DIR/manage-daemon.sh" status >/dev/null 2>&1; then
        success "Daemon started successfully"
    else
        warning "Daemon may not have started correctly. Check logs with: $CONFIG_DIR/manage-daemon.sh logs"
    fi
}

# Verify installation
verify_installation() {
    info "Verifying installation..."
    
    local failed=0
    
    # Check daemon
    if [[ -x "$DAEMON_TARGET" ]]; then
        success "✓ Daemon binary installed"
    else
        error "✗ Daemon binary not found"
        failed=1
    fi
    
    # Check Python packages
    if command -v workspace-qdrant-mcp >/dev/null 2>&1; then
        success "✓ MCP server command available"
    else
        error "✗ MCP server command not found"
        failed=1
    fi
    
    if command -v wqm >/dev/null 2>&1; then
        success "✓ WQM CLI command available"
    else
        error "✗ WQM CLI command not found"
        failed=1
    fi
    
    # Check configuration
    if [[ -f "$CONFIG_DIR/config/default.yaml" ]]; then
        success "✓ Default configuration created"
    else
        error "✗ Default configuration missing"
        failed=1
    fi
    
    # Check daemon status
    if "$CONFIG_DIR/manage-daemon.sh" status >/dev/null 2>&1; then
        success "✓ Daemon is running"
    else
        warning "⚠ Daemon is not running (this may be expected)"
    fi
    
    if (( failed == 0 )); then
        success "Installation verification completed successfully!"
        return 0
    else
        error "Installation verification failed"
        return 1
    fi
}

# Print usage information
print_usage_info() {
    cat << EOF

${GREEN}Workspace Qdrant MCP Installation Complete!${NC}

${BLUE}Available Commands:${NC}
  workspace-qdrant-mcp    - Start MCP server
  wqm                     - Workspace Qdrant Manager CLI
  $DAEMON_TARGET          - Rust daemon binary

${BLUE}Configuration:${NC}
  Config directory: $CONFIG_DIR
  Default config: $CONFIG_DIR/config/default.yaml
  MCP config template: $CONFIG_DIR/config/mcp.json

${BLUE}Daemon Management:${NC}
  Start:   $CONFIG_DIR/manage-daemon.sh start
  Stop:    $CONFIG_DIR/manage-daemon.sh stop
  Status:  $CONFIG_DIR/manage-daemon.sh status
  Logs:    $CONFIG_DIR/manage-daemon.sh logs

${BLUE}Next Steps:${NC}
  1. Start Qdrant: docker run -p 6333:6333 qdrant/qdrant
  2. Test server: workspace-qdrant-mcp --help
  3. Configure: edit $CONFIG_DIR/config/default.yaml
  4. For Claude: copy $CONFIG_DIR/config/mcp.json to your Claude config

${BLUE}Logs:${NC}
  Installation log: $INSTALL_LOG
  Daemon logs: $CONFIG_DIR/logs/memexd.log
  Server logs: $CONFIG_DIR/logs/workspace-qdrant-mcp.log

EOF
}

# Main installation flow
main() {
    info "Starting Workspace Qdrant MCP installation..."
    info "Platform: $(detect_platform)"
    info "Log file: $INSTALL_LOG"
    
    check_prerequisites
    install_daemon
    install_server
    create_directories
    generate_config
    create_mcp_config
    create_daemon_manager
    start_daemon
    verify_installation
    
    success "Installation completed successfully!"
    print_usage_info
}

# Handle script interruption
trap 'error "Installation interrupted"' INT TERM

# Check if script is being run directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi