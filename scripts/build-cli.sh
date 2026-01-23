#!/usr/bin/env bash
#
# Build script for wqm Rust CLI
#
# Usage:
#   ./scripts/build-cli.sh              # Build for current platform
#   ./scripts/build-cli.sh --all        # Build for all platforms (requires cross)
#   ./scripts/build-cli.sh --target <t> # Build for specific target
#   ./scripts/build-cli.sh --install    # Build and install to ~/.local/bin
#
# Supported targets:
#   x86_64-apple-darwin     - macOS Intel
#   aarch64-apple-darwin    - macOS Apple Silicon
#   x86_64-unknown-linux-gnu - Linux x64
#   x86_64-pc-windows-msvc  - Windows x64

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CLI_DIR="$PROJECT_ROOT/src/rust/cli"
OUTPUT_DIR="$PROJECT_ROOT/dist"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Detect current platform
detect_platform() {
    case "$(uname -s)" in
        Darwin)
            case "$(uname -m)" in
                x86_64) echo "x86_64-apple-darwin" ;;
                arm64)  echo "aarch64-apple-darwin" ;;
            esac
            ;;
        Linux)
            case "$(uname -m)" in
                x86_64) echo "x86_64-unknown-linux-gnu" ;;
                aarch64) echo "aarch64-unknown-linux-gnu" ;;
            esac
            ;;
        MINGW*|MSYS*|CYGWIN*)
            echo "x86_64-pc-windows-msvc"
            ;;
        *)
            error "Unsupported platform: $(uname -s)"
            exit 1
            ;;
    esac
}

# Build for a specific target
build_target() {
    local target="$1"
    local use_cross="$2"

    info "Building for target: $target"

    cd "$CLI_DIR"

    if [[ "$use_cross" == "true" ]] && command -v cross &> /dev/null; then
        info "Using cross for cross-compilation"
        cross build --release --target "$target"
    else
        cargo build --release --target "$target"
    fi

    # Determine binary name
    local binary_name="wqm"
    if [[ "$target" == *"windows"* ]]; then
        binary_name="wqm.exe"
    fi

    # Check binary exists
    local binary_path="$CLI_DIR/target/$target/release/$binary_name"
    if [[ -f "$binary_path" ]]; then
        info "Built: $binary_path"

        # Show binary size
        local size=$(ls -lh "$binary_path" | awk '{print $5}')
        info "Binary size: $size"

        # Strip on Unix platforms
        if [[ "$target" != *"windows"* ]] && command -v strip &> /dev/null; then
            info "Stripping binary..."
            strip "$binary_path"
            size=$(ls -lh "$binary_path" | awk '{print $5}')
            info "Stripped size: $size"
        fi

        return 0
    else
        error "Binary not found: $binary_path"
        return 1
    fi
}

# Install binary to user's local bin
install_binary() {
    local target="$1"

    # Determine binary path
    local binary_name="wqm"
    if [[ "$target" == *"windows"* ]]; then
        binary_name="wqm.exe"
    fi

    local binary_path="$CLI_DIR/target/$target/release/$binary_name"
    local install_dir="$HOME/.local/bin"

    if [[ ! -f "$binary_path" ]]; then
        error "Binary not found. Build first with: $0"
        exit 1
    fi

    mkdir -p "$install_dir"

    info "Installing to $install_dir/$binary_name"
    cp "$binary_path" "$install_dir/$binary_name"
    chmod +x "$install_dir/$binary_name"

    # Verify installation
    if "$install_dir/$binary_name" --version &> /dev/null; then
        info "Installation successful!"
        "$install_dir/$binary_name" --version
    else
        error "Installation verification failed"
        exit 1
    fi

    # Check if in PATH
    if ! command -v wqm &> /dev/null; then
        warn "$install_dir is not in your PATH"
        warn "Add to your shell profile: export PATH=\"\$PATH:$install_dir\""
    fi
}

# Copy to distribution directory
copy_to_dist() {
    local target="$1"

    local binary_name="wqm"
    local ext=""
    if [[ "$target" == *"windows"* ]]; then
        binary_name="wqm.exe"
        ext=".exe"
    fi

    local binary_path="$CLI_DIR/target/$target/release/$binary_name"

    if [[ ! -f "$binary_path" ]]; then
        return 1
    fi

    mkdir -p "$OUTPUT_DIR"

    # Create platform-specific name
    local platform_name
    case "$target" in
        x86_64-apple-darwin)    platform_name="wqm-macos-intel$ext" ;;
        aarch64-apple-darwin)   platform_name="wqm-macos-arm64$ext" ;;
        x86_64-unknown-linux-gnu) platform_name="wqm-linux-x64$ext" ;;
        x86_64-pc-windows-msvc) platform_name="wqm-windows-x64$ext" ;;
        *) platform_name="wqm-$target$ext" ;;
    esac

    cp "$binary_path" "$OUTPUT_DIR/$platform_name"
    info "Copied to: $OUTPUT_DIR/$platform_name"
}

# Build all supported targets
build_all() {
    local targets=(
        "x86_64-apple-darwin"
        "aarch64-apple-darwin"
        "x86_64-unknown-linux-gnu"
        "x86_64-pc-windows-msvc"
    )

    info "Building for all targets..."

    # Check for cross
    if ! command -v cross &> /dev/null; then
        warn "cross not installed. Install with: cargo install cross"
        warn "Only building for native target"
        targets=("$(detect_platform)")
    fi

    local success=0
    local failed=0

    for target in "${targets[@]}"; do
        echo ""
        if build_target "$target" "true"; then
            copy_to_dist "$target"
            ((success++))
        else
            ((failed++))
        fi
    done

    echo ""
    info "Build summary: $success succeeded, $failed failed"

    if [[ $success -gt 0 ]]; then
        info "Binaries available in: $OUTPUT_DIR"
        ls -lh "$OUTPUT_DIR"
    fi
}

# Show help
show_help() {
    cat << EOF
Build script for wqm Rust CLI

Usage:
  $0                     Build for current platform
  $0 --all               Build for all platforms
  $0 --target <target>   Build for specific target
  $0 --install           Build and install to ~/.local/bin
  $0 --help              Show this help

Supported targets:
  x86_64-apple-darwin      macOS Intel
  aarch64-apple-darwin     macOS Apple Silicon
  x86_64-unknown-linux-gnu Linux x64
  x86_64-pc-windows-msvc   Windows x64

Examples:
  $0                              # Build for current platform
  $0 --install                    # Build and install locally
  $0 --target x86_64-apple-darwin # Build for macOS Intel
  $0 --all                        # Build all platforms (needs cross)

Requirements:
  - Rust toolchain (rustup)
  - protobuf compiler (protoc)
  - cross (for cross-compilation): cargo install cross

EOF
}

# Main
main() {
    case "${1:-}" in
        --help|-h)
            show_help
            ;;
        --all)
            build_all
            ;;
        --target)
            if [[ -z "${2:-}" ]]; then
                error "Missing target argument"
                exit 1
            fi
            build_target "$2" "false"
            copy_to_dist "$2"
            ;;
        --install)
            local target
            target=$(detect_platform)
            build_target "$target" "false"
            install_binary "$target"
            ;;
        "")
            local target
            target=$(detect_platform)
            info "Building for native platform: $target"
            build_target "$target" "false"
            copy_to_dist "$target"
            ;;
        *)
            error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
}

main "$@"
