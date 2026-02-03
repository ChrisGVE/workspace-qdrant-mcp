#!/usr/bin/env bash
#
# workspace-qdrant-mcp binary installer
#
# Downloads and installs pre-built binaries from GitHub releases.
# For source builds, use ./install.sh instead.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/ChrisGVE/workspace-qdrant-mcp/main/scripts/download-install.sh | bash
#
# Or download and run:
#   ./scripts/download-install.sh [OPTIONS]
#
# Options:
#   --prefix PATH      Installation prefix (default: ~/.local)
#   --version VERSION  Specific version to install (default: latest)
#   --cli-only         Install only CLI (skip daemon)
#   --help             Show this help message
#
# Environment variables:
#   INSTALL_PREFIX     Same as --prefix
#   WQM_VERSION        Same as --version
#

set -e

# Configuration
REPO="ChrisGVE/workspace-qdrant-mcp"
GITHUB_API="https://api.github.com/repos/$REPO"
GITHUB_RELEASES="https://github.com/$REPO/releases"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Defaults
DEFAULT_PREFIX="$HOME/.local"
INSTALL_PREFIX="${INSTALL_PREFIX:-$DEFAULT_PREFIX}"
VERSION="${WQM_VERSION:-latest}"
CLI_ONLY=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --prefix)
            INSTALL_PREFIX="$2"
            shift 2
            ;;
        --version)
            VERSION="$2"
            shift 2
            ;;
        --cli-only)
            CLI_ONLY=true
            shift
            ;;
        --help|-h)
            head -25 "$0" | tail -21
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

BIN_DIR="$INSTALL_PREFIX/bin"

# Helper functions
info() { echo -e "${BLUE}==>${NC} $1"; }
success() { echo -e "${GREEN}==>${NC} $1"; }
warn() { echo -e "${YELLOW}Warning:${NC} $1"; }
error() { echo -e "${RED}Error:${NC} $1"; exit 1; }

# Detect platform
detect_platform() {
    local os arch target

    case "$(uname -s)" in
        Linux)  os="linux" ;;
        Darwin) os="darwin" ;;
        MINGW*|MSYS*|CYGWIN*) os="windows" ;;
        *) error "Unsupported OS: $(uname -s)" ;;
    esac

    case "$(uname -m)" in
        x86_64|amd64) arch="x64" ;;
        aarch64|arm64) arch="arm64" ;;
        *) error "Unsupported architecture: $(uname -m)" ;;
    esac

    # Map to Rust target triple for individual binaries
    case "${os}-${arch}" in
        linux-x64)   target="x86_64-unknown-linux-gnu" ;;
        linux-arm64) target="aarch64-unknown-linux-gnu" ;;
        darwin-x64)  target="x86_64-apple-darwin" ;;
        darwin-arm64) target="aarch64-apple-darwin" ;;
        windows-x64) target="x86_64-pc-windows-msvc" ;;
        windows-arm64) target="aarch64-pc-windows-msvc" ;;
    esac

    echo "$os $arch $target"
}

# Get latest version from GitHub API
get_latest_version() {
    local latest
    latest=$(curl -fsSL "$GITHUB_API/releases/latest" 2>/dev/null | grep '"tag_name"' | sed -E 's/.*"tag_name": *"([^"]+)".*/\1/')

    if [[ -z "$latest" ]]; then
        error "Could not determine latest version. Check your internet connection or specify --version"
    fi

    echo "$latest"
}

# Verify checksum
verify_checksum() {
    local file="$1"
    local expected="$2"

    if command -v sha256sum &>/dev/null; then
        local actual=$(sha256sum "$file" | awk '{print $1}')
    elif command -v shasum &>/dev/null; then
        local actual=$(shasum -a 256 "$file" | awk '{print $1}')
    else
        warn "No sha256sum or shasum found, skipping checksum verification"
        return 0
    fi

    if [[ "$actual" != "$expected" ]]; then
        error "Checksum mismatch for $file\n  Expected: $expected\n  Actual:   $actual"
    fi

    success "Checksum verified for $(basename "$file")"
}

# Download binary
download_binary() {
    local name="$1"
    local target="$2"
    local version="$3"
    local dest="$4"

    local url="$GITHUB_RELEASES/download/$version/${name}-${target}"
    local checksum_url="${url}.sha256"

    info "Downloading $name for $target..."

    # Download binary
    if ! curl -fsSL "$url" -o "$dest/$name"; then
        error "Failed to download $name from $url"
    fi

    # Download and verify checksum
    local expected_checksum
    if expected_checksum=$(curl -fsSL "$checksum_url" 2>/dev/null); then
        verify_checksum "$dest/$name" "$expected_checksum"
    else
        warn "Checksum file not available, skipping verification"
    fi

    chmod 755 "$dest/$name"
}

# Main installation
main() {
    echo ""
    echo "workspace-qdrant-mcp binary installer"
    echo "======================================"
    echo ""

    # Detect platform
    read -r os arch target <<< "$(detect_platform)"
    info "Detected platform: $os $arch ($target)"

    # Resolve version
    if [[ "$VERSION" == "latest" ]]; then
        info "Fetching latest version..."
        VERSION=$(get_latest_version)
    fi
    info "Installing version: $VERSION"

    # Create directories
    mkdir -p "$BIN_DIR"

    # Create temp directory
    TEMP_DIR=$(mktemp -d)
    trap "rm -rf $TEMP_DIR" EXIT

    # Download CLI
    download_binary "wqm" "$target" "$VERSION" "$TEMP_DIR"

    # Download daemon (unless --cli-only)
    if [[ "$CLI_ONLY" == false ]]; then
        download_binary "memexd" "$target" "$VERSION" "$TEMP_DIR"
    fi

    # Install binaries
    info "Installing to $BIN_DIR..."
    mv "$TEMP_DIR/wqm" "$BIN_DIR/"
    success "Installed wqm"

    if [[ "$CLI_ONLY" == false ]] && [[ -f "$TEMP_DIR/memexd" ]]; then
        mv "$TEMP_DIR/memexd" "$BIN_DIR/"
        success "Installed memexd"
    fi

    # Verify installation
    if [[ -x "$BIN_DIR/wqm" ]]; then
        WQM_VERSION=$("$BIN_DIR/wqm" --version 2>/dev/null || echo "unknown")
        success "wqm version: $WQM_VERSION"
    fi

    if [[ -x "$BIN_DIR/memexd" ]]; then
        MEMEXD_VERSION=$("$BIN_DIR/memexd" --version 2>/dev/null || echo "unknown")
        success "memexd version: $MEMEXD_VERSION"
    fi

    # PATH check
    if [[ ":$PATH:" != *":$BIN_DIR:"* ]]; then
        warn "$BIN_DIR is not in your PATH"
        echo ""
        echo "Add to your shell profile:"
        echo "  export PATH=\"$BIN_DIR:\$PATH\""
        echo ""
    fi

    echo ""
    echo "======================================"
    success "Installation complete!"
    echo "======================================"
    echo ""
    echo "Quick start:"
    echo "  1. Start Qdrant: docker run -p 6333:6333 qdrant/qdrant"
    echo "  2. Check health: wqm admin health"
    echo "  3. View help: wqm --help"
    echo ""

    if [[ "$CLI_ONLY" == false ]] && [[ -x "$BIN_DIR/memexd" ]]; then
        echo "To start the daemon service:"
        echo "  wqm service install"
        echo "  wqm service start"
        echo ""
    fi
}

main
