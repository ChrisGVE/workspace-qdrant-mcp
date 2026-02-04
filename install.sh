#!/usr/bin/env bash
#
# workspace-qdrant-mcp installer
#
# Builds and installs the CLI (wqm), daemon (memexd), and Python MCP server.
# All binaries are self-contained with ONNX Runtime statically linked.
#
# Usage:
#   ./install.sh [OPTIONS]
#
# Options:
#   --prefix PATH      Installation prefix (default: ~/.local)
#   --force            Clean rebuild from scratch (cargo clean)
#   --no-service       Skip daemon service installation
#   --no-verify        Skip verification steps
#   --cli-only         Build only CLI (skip daemon)
#   --help             Show this help message
#
# Environment variables:
#   INSTALL_PREFIX     Same as --prefix
#   BIN_DIR            Override binary installation directory
#   ORT_LIB_LOCATION   Override ONNX Runtime location (Intel Mac only)
#
# Examples:
#   ./install.sh                          # Install to ~/.local/bin
#   ./install.sh --prefix /usr/local      # Install to /usr/local/bin
#   ./install.sh --force                  # Clean rebuild from scratch
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
FORCE=false

# ONNX Runtime version for Intel Mac (static library from supertone-inc/onnxruntime-build)
ORT_VERSION="1.23.2"
ORT_CACHE_DIR="$HOME/.onnxruntime-static"

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
        --force)
            FORCE=true
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

# Detect platform
detect_platform() {
    OS="$(uname -s)"
    ARCH="$(uname -m)"

    case "$OS" in
        Darwin)
            if [[ "$ARCH" == "x86_64" ]]; then
                PLATFORM="intel-mac"
            else
                PLATFORM="arm-mac"
            fi
            ;;
        Linux)
            PLATFORM="linux"
            ;;
        *)
            PLATFORM="other"
            ;;
    esac

    info "Detected platform: $OS $ARCH ($PLATFORM)"
}

# Download ONNX Runtime static library for Intel Mac
# Uses supertone-inc/onnxruntime-build which provides static libraries (.a)
# The official Microsoft releases only include dynamic libraries (.dylib)
setup_onnx_runtime() {
    if [[ "$PLATFORM" != "intel-mac" ]]; then
        return 0
    fi

    # Check if ORT_LIB_LOCATION is already set
    if [[ -n "$ORT_LIB_LOCATION" ]]; then
        info "Using existing ORT_LIB_LOCATION: $ORT_LIB_LOCATION"
        return 0
    fi

    info "Intel Mac detected: ONNX Runtime static library download required"
    info "The ort crate dropped x86_64-apple-darwin support in v2.0.0-rc.11"

    # Check if already downloaded (look for static library)
    if [[ -f "$ORT_CACHE_DIR/lib/libonnxruntime.a" ]]; then
        info "Found cached ONNX Runtime static library at $ORT_CACHE_DIR"
        export ORT_LIB_LOCATION="$ORT_CACHE_DIR/lib"
        return 0
    fi

    info "Downloading ONNX Runtime $ORT_VERSION static library for Intel Mac..."
    mkdir -p "$ORT_CACHE_DIR"

    # Download from supertone-inc/onnxruntime-build (provides static libraries)
    # Universal2 includes both x86_64 and ARM64 architectures
    ORT_URL="https://github.com/supertone-inc/onnxruntime-build/releases/download/v${ORT_VERSION}/onnxruntime-osx-universal2-static_lib-${ORT_VERSION}.tgz"
    ORT_TMP="$ORT_CACHE_DIR/onnxruntime-static.tgz"

    info "Downloading from supertone-inc/onnxruntime-build..."
    if command -v curl &> /dev/null; then
        curl -L -o "$ORT_TMP" "$ORT_URL"
    elif command -v wget &> /dev/null; then
        wget -q -O "$ORT_TMP" "$ORT_URL"
    else
        error "Neither curl nor wget found. Please install one of them."
    fi

    # Extract (archive structure: ./lib/libonnxruntime.a, ./include/...)
    tar xzf "$ORT_TMP" -C "$ORT_CACHE_DIR"
    rm -f "$ORT_TMP"

    if [[ -f "$ORT_CACHE_DIR/lib/libonnxruntime.a" ]]; then
        success "ONNX Runtime static library downloaded to $ORT_CACHE_DIR"
        # Point to lib subdirectory where libonnxruntime.a is located
        export ORT_LIB_LOCATION="$ORT_CACHE_DIR/lib"
    else
        error "Failed to download ONNX Runtime static library"
    fi
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

    if [ "$FORCE" = true ]; then
        info "Force rebuild: cleaning previous build artifacts..."
        cargo clean
    fi

    # Configure ONNX Runtime linking for Intel Mac
    if [[ "$PLATFORM" == "intel-mac" ]] && [[ -n "$ORT_LIB_LOCATION" ]]; then
        info "ORT_LIB_LOCATION set to: $ORT_LIB_LOCATION"
        info "Intel Mac: Using static library for self-contained binary"
    fi

    if [ "$CLI_ONLY" = true ]; then
        info "Building CLI only (--cli-only specified)..."
        cargo build --release -p wqm-cli
    else
        info "Building CLI..."
        cargo build --release -p wqm-cli

        info "Building daemon..."
        if cargo build --release -p memexd; then
            success "Daemon built successfully"
        else
            error "Daemon build failed. Check error messages above."
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

        # Verify self-contained binary (macOS - all platforms use static linking now)
        if [[ "$PLATFORM" == "intel-mac" ]] || [[ "$PLATFORM" == "arm-mac" ]]; then
            if otool -L "$BIN_DIR/memexd" | grep -qi "libonnxruntime"; then
                warn "Binary appears to have external ONNX Runtime dependency"
            else
                success "Binary is self-contained (ONNX Runtime statically linked)"
            fi
        fi
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
        echo "    (self-contained binary with ONNX Runtime statically linked)"
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
    echo "  2. Start daemon: $BIN_DIR/memexd &"
    echo "  3. Run MCP server: uv run workspace-qdrant-mcp"
    echo "  4. Use CLI: wqm --help"
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

    detect_platform
    check_prerequisites
    setup_onnx_runtime
    create_directories
    build_rust
    install_binaries
    install_python
    verify_installation
    setup_service
    print_summary
}

main
