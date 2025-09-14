#!/bin/bash
# Task 215: Commit logging migration progress

set -e

cd /Users/chris/Dropbox/dev/ai/claude-code-cfg/mcp/workspace-qdrant-mcp

echo "Task 215: Committing logging migration progress..."

# Add migrated files
git add src/python/common/logging.py
git add src/python/workspace_qdrant_mcp/stdio_server.py
git add src/python/common/core/hybrid_search.py

# Commit with Task 215 reference
git commit -m "feat(logging): Task 215 - migrate critical files to unified logging system

- Create common/logging.py bridge module for unified logging access
- Migrate stdio_server.py to use get_logger() instead of direct logging
- Replace sys.__stderr__.write() with safe_log_error() for stdio compliance
- Migrate hybrid_search.py to use unified logging system
- Add structured logging with MCP stdio mode detection
- Maintain full compatibility with existing logging configuration

Migration Status:
✓ stdio_server.py - critical sys.__stderr__ usage eliminated
✓ hybrid_search.py - direct logging.getLogger() calls replaced
✓ common/logging.py - unified bridge module created

Remaining: ~111 files requiring direct logging migration"