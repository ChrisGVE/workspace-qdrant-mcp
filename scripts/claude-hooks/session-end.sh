#!/usr/bin/env bash
# Claude Code session-end hook for workspace-qdrant-mcp
# Notifies daemon when Claude Code session ends

set -euo pipefail

# Configuration
MCP_ENDPOINT="${MCP_ENDPOINT:-http://localhost:8765/api/v1/hooks/session-end}"
LOG_FILE="${HOME}/.claude/hooks.log"
SESSION_ID="${CLAUDE_SESSION_ID:-unknown}"
REASON="${1:-other}"

# Create log directory if needed
mkdir -p "$(dirname "$LOG_FILE")"

# Construct JSON payload
PAYLOAD=$(cat <<EOF
{
  "session_id": "${SESSION_ID}",
  "reason": "${REASON}"
}
EOF
)

# Send request asynchronously to avoid blocking Claude Code
{
  echo "[$(date -Iseconds)] session-end: Sending hook to MCP endpoint (reason: ${REASON})" >> "$LOG_FILE"
  echo "[$(date -Iseconds)] session-end: Session ID: ${SESSION_ID}" >> "$LOG_FILE"

  if curl -s -X POST "${MCP_ENDPOINT}" \
    -H "Content-Type: application/json" \
    -d "${PAYLOAD}" \
    --max-time 5 \
    --connect-timeout 2 >> "$LOG_FILE" 2>&1; then
    echo "[$(date -Iseconds)] session-end: Hook sent successfully" >> "$LOG_FILE"
  else
    EXIT_CODE=$?
    echo "[$(date -Iseconds)] session-end: Failed to send hook (exit code: ${EXIT_CODE})" >> "$LOG_FILE"
  fi

  # Rotate log file if it exceeds 10MB
  LOG_SIZE=$(stat -f%z "$LOG_FILE" 2>/dev/null || stat -c%s "$LOG_FILE" 2>/dev/null || echo 0)
  if [ "$LOG_SIZE" -gt 10485760 ]; then
    mv "$LOG_FILE" "${LOG_FILE}.old"
    echo "[$(date -Iseconds)] Log file rotated" >> "$LOG_FILE"
  fi
} &

# Exit immediately to avoid blocking Claude Code
exit 0
