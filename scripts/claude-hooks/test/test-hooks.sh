#!/usr/bin/env bash
# Test script to validate Claude Code hook functionality

set -euo pipefail

echo "Testing Claude Code hook integration..."
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test configuration
TEST_SESSION_ID="test-session-$(date +%s)"
TEST_PROJECT_DIR="/tmp/test-project"
HOOK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Test Configuration:"
echo "  Session ID: ${TEST_SESSION_ID}"
echo "  Project Dir: ${TEST_PROJECT_DIR}"
echo "  Hook Scripts: ${HOOK_DIR}"
echo ""

# Check if MCP server is running
echo -n "Checking MCP server health... "
if curl -s -f http://localhost:8765/api/v1/health > /dev/null 2>&1; then
  echo -e "${GREEN}OK${NC}"
else
  echo -e "${YELLOW}WARNING${NC}: MCP server not responding"
  echo "  Please start the server: python -m workspace_qdrant_mcp.http_server"
  echo "  Or set MCP_ENDPOINT to a test server"
  echo ""
fi

# Test session-start hook
echo -n "Testing session-start hook... "
export CLAUDE_SESSION_ID="${TEST_SESSION_ID}"
export CLAUDE_PROJECT_DIR="${TEST_PROJECT_DIR}"

if bash "${HOOK_DIR}/session-start.sh" startup > /dev/null 2>&1; then
  echo -e "${GREEN}OK${NC}"
else
  echo -e "${RED}FAILED${NC}"
  exit 1
fi

# Wait for async request to complete
sleep 2

# Test session-end hook
echo -n "Testing session-end hook... "
if bash "${HOOK_DIR}/session-end.sh" other > /dev/null 2>&1; then
  echo -e "${GREEN}OK${NC}"
else
  echo -e "${RED}FAILED${NC}"
  exit 1
fi

# Wait for async request to complete
sleep 2

# Check log file
LOG_FILE="${HOME}/.claude/hooks.log"
echo ""
echo "Hook execution log (last 10 lines):"
echo "-----------------------------------"
if [ -f "$LOG_FILE" ]; then
  tail -10 "$LOG_FILE"
else
  echo -e "${YELLOW}No log file found at ${LOG_FILE}${NC}"
fi
echo "-----------------------------------"
echo ""

# Test different session-end reasons
echo "Testing different session-end reasons:"
for REASON in clear logout prompt_input_exit; do
  echo -n "  ${REASON}... "
  if bash "${HOOK_DIR}/session-end.sh" "$REASON" > /dev/null 2>&1; then
    echo -e "${GREEN}OK${NC}"
  else
    echo -e "${RED}FAILED${NC}"
  fi
  sleep 1
done

echo ""
echo -e "${GREEN}All tests passed!${NC}"
echo ""
echo "Next steps:"
echo "1. Check detailed logs: cat ${LOG_FILE}"
echo "2. Verify MCP server received requests"
echo "3. Install hooks in your project: bash scripts/install-claude-hooks.sh"
