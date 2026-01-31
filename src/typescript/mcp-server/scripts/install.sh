#!/usr/bin/env bash
#
# install-ts-server.sh - Install the workspace-qdrant-mcp TypeScript server
#
# This script builds and optionally installs the MCP server globally.
#
# Usage:
#   ./scripts/install.sh          # Build only
#   ./scripts/install.sh --link   # Build and npm link globally
#   ./scripts/install.sh --global # Build and install globally via npm
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check Node.js version
check_node_version() {
    if ! command -v node &> /dev/null; then
        log_error "Node.js is not installed. Please install Node.js 18 or later."
        exit 1
    fi

    local node_version
    node_version=$(node --version | sed 's/v//' | cut -d. -f1)

    if [ "$node_version" -lt 18 ]; then
        log_error "Node.js version must be 18 or later. Current version: $(node --version)"
        exit 1
    fi

    log_info "Node.js version: $(node --version)"
}

# Install dependencies
install_dependencies() {
    log_info "Installing dependencies..."
    cd "$PROJECT_DIR"
    npm ci --ignore-scripts || npm install --ignore-scripts
}

# Build the project
build_project() {
    log_info "Building TypeScript server..."
    cd "$PROJECT_DIR"
    npm run build

    if [ -f "dist/index.js" ]; then
        log_info "Build successful: dist/index.js"
    else
        log_error "Build failed: dist/index.js not found"
        exit 1
    fi
}

# Run tests
run_tests() {
    log_info "Running tests..."
    cd "$PROJECT_DIR"
    npm test
}

# Link globally using npm link
link_global() {
    log_info "Linking globally with npm link..."
    cd "$PROJECT_DIR"
    npm link

    # Verify the link
    if command -v workspace-qdrant-mcp &> /dev/null; then
        log_info "Successfully linked: workspace-qdrant-mcp is now available globally"
        log_info "Location: $(which workspace-qdrant-mcp)"
    else
        log_warn "Link created but command not found in PATH. You may need to restart your terminal."
    fi
}

# Install globally
install_global() {
    log_info "Installing globally..."
    cd "$PROJECT_DIR"
    npm install -g .

    # Verify the installation
    if command -v workspace-qdrant-mcp &> /dev/null; then
        log_info "Successfully installed: workspace-qdrant-mcp is now available globally"
        log_info "Location: $(which workspace-qdrant-mcp)"
    else
        log_warn "Installation completed but command not found in PATH. You may need to restart your terminal."
    fi
}

# Main
main() {
    local mode="build"

    while [[ $# -gt 0 ]]; do
        case $1 in
            --link)
                mode="link"
                shift
                ;;
            --global)
                mode="global"
                shift
                ;;
            --help|-h)
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  --link    Build and npm link globally (for development)"
                echo "  --global  Build and install globally via npm"
                echo "  --help    Show this help message"
                echo ""
                echo "Without options, only builds the project."
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done

    log_info "workspace-qdrant-mcp TypeScript Server Installation"
    echo ""

    check_node_version
    install_dependencies
    build_project

    case $mode in
        link)
            link_global
            ;;
        global)
            run_tests
            install_global
            ;;
        build)
            log_info "Build complete. Use --link or --global to install."
            ;;
    esac

    echo ""
    log_info "Done!"
}

main "$@"
