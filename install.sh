#!/usr/bin/env bash
#
# workspace-qdrant-mcp installer
#
# Builds and installs the CLI (wqm), daemon (memexd), and Python MCP server.
#
# Usage:
#   ./install.sh [OPTIONS]
#
# Options:
#   --prefix PATH      Installation prefix (default: ~/.local)
#   --no-service       Skip daemon service installation
#   --no-verify        Skip verification steps
#   --cli-only         Build only CLI (skip daemon)
#   --help             Show this help message
#
# Environment variables:
#   INSTALL_PREFIX     Same as --prefix
#   BIN_DIR            Override binary installation directory
#
# Examples:
#   ./install.sh                          # Install to ~/.local/bin
#   ./install.sh --prefix /usr/local      # Install to /usr/local/bin
#   INSTALL_PREFIX=/opt/wqm ./install.sh  # Install to /opt/wqm/bin
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
DEFAULT_PREFIX="$HOME/.local"
INSTALL_PREFIX="${INSTALL_PREFIX:-$DEFAULT_PREFIX}"
NO_SERVICE=false
NO_VERIFY=false
CLI_ONLY=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --prefix)
            INSTALL_PREFIX="$2"
            shift 2
            ;;
        --no-service)
            NO_SERVICE=true
            shift
            ;;
        --no-verify)
            NO_VERIFY=true
            shift
            ;;
        --cli-only)
            CLI_ONLY=true
            shift
            ;;
        --help|-h)
            head -30 "$0" | tail -26
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

BIN_DIR="${BIN_DIR:-$INSTALL_PREFIX/bin}"

# Helper functions
info() {
    echo -e "${BLUE}==>${NC} $1"
}

success() {
    echo -e "${GREEN}==>${NC} $1"
}

warn() {
    echo -e "${YELLOW}Warning:${NC} $1"
}

error() {
    echo -e "${RED}Error:${NC} $1"
    exit 1
}

# Check prerequisites
check_prerequisites() {
    info "Checking prerequisites..."

    if ! command -v cargo &> /dev/null; then
        error "cargo not found. Please install Rust toolchain: https://rustup.rs"
    fi

    if ! command -v uv &> /dev/null; then
        error "uv not found. Please install uv: https://github.com/astral-sh/uv"
    fi

    success "Prerequisites OK (cargo: $(cargo --version | cut -d' ' -f2), uv: $(uv --version | cut -d' ' -f2))"
}

# Create directories
create_directories() {
    info "Creating directories..."
    mkdir -p "$BIN_DIR"
    success "Created $BIN_DIR"
}

# Build Rust binaries
build_rust() {
    info "Building Rust binaries from unified workspace..."

    cd src/rust

    if [ "$CLI_ONLY" = true ]; then
        info "Building CLI only (--cli-only specified)..."
        cargo build --release -p wqm-cli
    else
        # Try to build both, but CLI might succeed while daemon fails (ort issue on Intel Mac)
        info "Building CLI..."
        cargo build --release -p wqm-cli

        info "Attempting to build daemon..."
        if cargo build --release -p memexd 2>/dev/null; then
            success "Daemon built successfully"
        else
            warn "Daemon build failed (common on Intel Mac due to ONNX Runtime)"
            warn "CLI will still be installed. For daemon, see docs/TROUBLESHOOTING.md"
            CLI_ONLY=true
        fi
    fi

    cd ../..
    success "Rust build complete"
}

# Install binaries
install_binaries() {
    info "Installing binaries to $BIN_DIR..."

    # CLI
    cp src/rust/target/release/wqm "$BIN_DIR/"
    chmod 755 "$BIN_DIR/wqm"
    success "Installed wqm"

    # Daemon (if built)
    if [ "$CLI_ONLY" = false ] && [ -f src/rust/target/release/memexd ]; then
        cp src/rust/target/release/memexd "$BIN_DIR/"
        chmod 755 "$BIN_DIR/memexd"
        success "Installed memexd"
    fi
}

# Install Python components
install_python() {
    info "Installing Python MCP server..."
    uv sync
    success "Python dependencies installed"
}

# Verify installation
verify_installation() {
    if [ "$NO_VERIFY" = true ]; then
        return 0
    fi

    info "Verifying installation..."

    # Check if BIN_DIR is in PATH
    if [[ ":$PATH:" != *":$BIN_DIR:"* ]]; then
        warn "$BIN_DIR is not in your PATH"
        echo "  Add to your shell profile:"
        echo "    export PATH=\"$BIN_DIR:\$PATH\""
    fi

    # Test wqm
    if [ -f "$BIN_DIR/wqm" ]; then
        WQM_VERSION=$("$BIN_DIR/wqm" --version 2>/dev/null || echo "unknown")
        success "wqm version: $WQM_VERSION"
    else
        warn "wqm binary not found at $BIN_DIR/wqm"
    fi

    # Test memexd
    if [ -f "$BIN_DIR/memexd" ]; then
        MEMEXD_VERSION=$("$BIN_DIR/memexd" --version 2>/dev/null || echo "unknown")
        success "memexd version: $MEMEXD_VERSION"
    fi

    # Test Python server
    if uv run workspace-qdrant-mcp --help &>/dev/null; then
        success "MCP server ready"
    else
        warn "MCP server not responding (may need Qdrant running)"
    fi
}

# Setup daemon service
setup_service() {
    if [ "$NO_SERVICE" = true ] || [ "$CLI_ONLY" = true ]; then
        return 0
    fi

    if [ ! -f "$BIN_DIR/memexd" ]; then
        return 0
    fi

    info "Setting up daemon service..."
    echo ""
    echo "To install and start the daemon service, run:"
    echo "  $BIN_DIR/wqm service install"
    echo "  $BIN_DIR/wqm service start"
    echo ""
}

# Print summary
print_summary() {
    echo ""
    echo "======================================"
    success "Installation complete!"
    echo "======================================"
    echo ""
    echo "Installed components:"
    echo "  - wqm (CLI): $BIN_DIR/wqm"
    if [ -f "$BIN_DIR/memexd" ]; then
        echo "  - memexd (daemon): $BIN_DIR/memexd"
    fi
    echo "  - MCP server: uv run workspace-qdrant-mcp"
    echo ""

    if [[ ":$PATH:" != *":$BIN_DIR:"* ]]; then
        echo "Add to your PATH:"
        echo "  export PATH=\"$BIN_DIR:\$PATH\""
        echo ""
    fi

    echo "Quick start:"
    echo "  1. Start Qdrant: docker run -p 6333:6333 qdrant/qdrant"
    echo "  2. Run MCP server: uv run workspace-qdrant-mcp"
    echo "  3. Use CLI: wqm --help"
    echo ""
}

# Main installation flow
main() {
    echo ""
    echo "workspace-qdrant-mcp installer"
    echo "=============================="
    echo ""
    echo "Installation prefix: $INSTALL_PREFIX"
    echo "Binary directory: $BIN_DIR"
    echo ""

    check_prerequisites
    create_directories
    build_rust
    install_binaries
    install_python
    verify_installation
    setup_service
    print_summary
}

main
