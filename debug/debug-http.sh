#!/bin/bash
# Debug workspace-qdrant-mcp server using MCP Inspector (HTTP mode)
#
# This script starts the MCP Inspector UI for debugging the workspace-qdrant-mcp
# server using HTTP transport. You need to start the HTTP server separately.
#
# 1. Start the HTTP server: uv run python -m workspace_qdrant_mcp.server --transport http --port 8000
# 2. Run this script to open the inspector

set -e

echo "Starting MCP Inspector for workspace-qdrant-mcp (HTTP mode)..."
echo "Inspector UI will be available at: http://localhost:6274"
echo ""
echo "NOTE: Make sure the HTTP server is running on port 8000:"
echo "  uv run python -m workspace_qdrant_mcp.server --transport http --port 8000"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Ensure we're in the debug directory
cd "$(dirname "$0")"

# Run MCP Inspector with HTTP configuration
npm run debug-http