#!/bin/bash
# Debug workspace-qdrant-mcp server using MCP Inspector (STDIO mode)
#
# This script starts the MCP Inspector UI for debugging the workspace-qdrant-mcp
# server using STDIO transport. The inspector will be available at:
# http://localhost:6274

set -e

echo "Starting MCP Inspector for workspace-qdrant-mcp (STDIO mode)..."
echo "Inspector UI will be available at: http://localhost:6274"
echo "Press Ctrl+C to stop"
echo ""

# Ensure we're in the debug directory
cd "$(dirname "$0")"

# Run MCP Inspector with STDIO configuration
npm run debug-stdio