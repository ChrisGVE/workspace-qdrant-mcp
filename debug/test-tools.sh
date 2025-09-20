#!/bin/bash
# Test workspace-qdrant-mcp server tools using MCP Inspector CLI
#
# This script demonstrates how to test individual MCP tools using the
# command-line interface of MCP Inspector.

set -e

echo "Testing workspace-qdrant-mcp server tools..."
echo "============================================"
echo ""

# Ensure we're in the debug directory
cd "$(dirname "$0")"

echo "1. Listing available tools..."
npm run list-tools
echo ""

echo "2. Listing available resources..."
npm run list-resources
echo ""

echo "3. Testing connection..."
npm run test-connection
echo ""

echo "Testing completed successfully!"