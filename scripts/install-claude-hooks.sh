#!/usr/bin/env bash
# Installation script for Claude Code hooks

set -euo pipefail

PROJECT_ROOT="${1:-.}"

echo "Installing Claude Code hooks for workspace-qdrant-mcp..."
echo ""

# Create directories
echo "Creating directories..."
mkdir -p "${PROJECT_ROOT}/.claude/hooks"
mkdir -p "${PROJECT_ROOT}/.claude/hooks/test"

# Copy hook scripts
echo "Copying hook scripts..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cp "${SCRIPT_DIR}/claude-hooks/session-start.sh" "${PROJECT_ROOT}/.claude/hooks/"
cp "${SCRIPT_DIR}/claude-hooks/session-end.sh" "${PROJECT_ROOT}/.claude/hooks/"
cp "${SCRIPT_DIR}/claude-hooks/test/test-hooks.sh" "${PROJECT_ROOT}/.claude/hooks/test/"

# Make executable
echo "Setting executable permissions..."
chmod +x "${PROJECT_ROOT}/.claude/hooks"/*.sh
chmod +x "${PROJECT_ROOT}/.claude/hooks/test"/*.sh

# Create settings.json if it doesn't exist
SETTINGS_FILE="${PROJECT_ROOT}/.claude/settings.json"
if [ ! -f "$SETTINGS_FILE" ]; then
  echo "Creating Claude Code settings file..."
  cat > "$SETTINGS_FILE" <<'EOF'
{
  "hooks": {
    "session-start": [
      {
        "matcher": ".*",
        "hooks": [".claude/hooks/session-start.sh"]
      }
    ],
    "session-end": [
      {
        "matcher": ".*",
        "hooks": [".claude/hooks/session-end.sh"]
      }
    ]
  }
}
EOF
else
  echo "WARNING: ${SETTINGS_FILE} already exists"
  echo "  Please manually add hooks configuration (see docs/claude_code_integration.md)"
fi

echo ""
echo "✓ Hooks installed successfully!"
echo ""
echo "Next steps:"
echo "1. Start MCP HTTP server:"
echo "   python -m workspace_qdrant_mcp.http_server"
echo ""
echo "2. Test hooks:"
echo "   bash ${PROJECT_ROOT}/.claude/hooks/test/test-hooks.sh"
echo ""
echo "3. Start using Claude Code in this project"
echo "   Hooks will automatically trigger on session start/end"
echo ""
echo "For more information, see: docs/claude_code_integration.md"
