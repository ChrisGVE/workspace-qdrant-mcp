#!/usr/bin/env bash
# Claude Code session-start hook for workspace-qdrant-mcp
# Triggers memory collection ingestion when a Claude Code session starts

set -euo pipefail

# Configuration
MCP_ENDPOINT="${MCP_ENDPOINT:-http://localhost:8765/api/v1/hooks/session-start}"
LOG_FILE="${HOME}/.claude/hooks.log"
SESSION_ID="${CLAUDE_SESSION_ID:-unknown}"
PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(pwd)}"
SOURCE="${1:-startup}"

# Create log directory if needed
mkdir -p "$(dirname "$LOG_FILE")"

# Construct JSON payload
PAYLOAD=$(cat <<EOF
{
  "session_id": "${SESSION_ID}",
  "project_dir": "${PROJECT_DIR}",
  "source": "${SOURCE}"
}
EOF
)

# Send request asynchronously to avoid blocking Claude Code
{
  echo "[$(date -Iseconds)] session-start: Sending hook to MCP endpoint" >> "$LOG_FILE"
  echo "[$(date -Iseconds)] session-start: Session ID: ${SESSION_ID}, Project: ${PROJECT_DIR}, Source: ${SOURCE}" >> "$LOG_FILE"

  if curl -s -X POST "${MCP_ENDPOINT}" \
    -H "Content-Type: application/json" \
    -d "${PAYLOAD}" \
    --max-time 5 \
    --connect-timeout 2 >> "$LOG_FILE" 2>&1; then
    echo "[$(date -Iseconds)] session-start: Hook sent successfully" >> "$LOG_FILE"
  else
    EXIT_CODE=$?
    echo "[$(date -Iseconds)] session-start: Failed to send hook (exit code: ${EXIT_CODE})" >> "$LOG_FILE"
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
