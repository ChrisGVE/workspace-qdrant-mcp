#!/bin/bash

# Simple global installation script for workspace-qdrant-mcp
set -e

echo "Installing workspace-qdrant-mcp globally..."

# Check prerequisites
if ! command -v uv &> /dev/null; then
    echo "Error: uv not found. Please install uv first."
    exit 1
fi

if ! command -v cargo &> /dev/null; then
    echo "Error: cargo not found. Please install Rust toolchain first."
    exit 1
fi

# Install Python components (server + wqm) globally for user
echo "Installing Python components via uv..."
uv tool install .

# Build and install Rust daemon
echo "Building Rust daemon..."
cd rust-engine-legacy
cargo build --release --bin memexd

# Install daemon to /usr/local/bin (no sudo needed on macOS)
echo "Installing daemon to /usr/local/bin..."
cp target/release/memexd /usr/local/bin/memexd
chmod +x /usr/local/bin/memexd

echo "Installation complete!"
echo ""
echo "Available commands:"
echo "  workspace-qdrant-mcp  - MCP server"
echo "  wqm                   - CLI tool"  
echo "  memexd                - Daemon"
echo ""
echo "To install and start daemon service:"
echo "  wqm service install"
echo "  wqm service start"