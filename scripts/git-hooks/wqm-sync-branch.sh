#!/bin/sh
# wqm-sync-branch.sh — sync the current git branch/worktree with workspace-qdrant MCP.
#
# Runs on the host where `git` is executed (Git Bash on Windows, sh on macOS/Linux).
# Calls the dockerized MCP HTTP server, which forwards the registration to the
# daemon via gRPC. Designed to be invoked from git hooks (post-checkout,
# post-commit, post-merge, post-rewrite, post-worktree-add).
#
# Always exits 0 so it never blocks the git operation, even on failure.
#
# Required: sh, git, curl
#
# Environment overrides:
#   WQM_MCP_URL       MCP HTTP endpoint   default: http://localhost:6335/mcp
#   WQM_MCP_TOKEN     Bearer token        default: $MCP_HTTP_TOKEN
#   WQM_HOOK_TIMEOUT  curl --max-time (s) default: 35
#   WQM_HOOK_LOG      Append log path     default: unset (silent)
#   WQM_HOOK_NAME     Override hook label default: $1 if passed, else "manual"
#
# The default 35s timeout covers `register_project` calls that trigger LSP
# server startup on the daemon side (pyright's `initialize` regularly takes
# ~10s before timing out, with other languages queued behind it). Drop this
# only when targeting a daemon configured to skip eager LSP startup.

set -u
# No -e: we silently swallow errors so git operations never break.

# Source shared path/git helpers.
_SELF_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$_SELF_DIR/../lib/path-resolver.sh"

MCP_URL="${WQM_MCP_URL:-http://localhost:6335/mcp}"
MCP_TOKEN="${WQM_MCP_TOKEN:-${MCP_HTTP_TOKEN:-}}"
TIMEOUT="${WQM_HOOK_TIMEOUT:-35}"
LOG_FILE="${WQM_HOOK_LOG:-}"
HOOK_NAME="${WQM_HOOK_NAME:-${1:-manual}}"

# Bail out silently if prerequisites missing.
wqm_path_has_tool git || exit 0
wqm_path_has_tool curl || exit 0

REPO_ROOT="$(wqm_git_repo_root "$PWD")"
[ -n "$REPO_ROOT" ] || exit 0

BRANCH="$(wqm_git_current_branch "$REPO_ROOT")"
[ -n "$BRANCH" ] || BRANCH="HEAD"
COMMIT="$(wqm_git_head_commit "$REPO_ROOT")"
REMOTE="$(wqm_git_remote_url "$REPO_ROOT")"
IS_WORKTREE="$(wqm_git_is_worktree "$REPO_ROOT")"

if [ "$IS_WORKTREE" = "true" ]; then
  WORKTREE_PATH="$REPO_ROOT"
else
  WORKTREE_PATH=""
fi

# Translate host paths to container-visible paths when the daemon runs in
# Docker. Requires both WQM_HOST_DEV_ROOT (path as seen by this hook on the
# host) and WQM_DEV_ROOT (path as bind-mounted inside the container) to be
# set; otherwise the original path is passed through unchanged.
_translate_to_container() {
  _p="$1"
  if [ -z "${WQM_HOST_DEV_ROOT:-}" ] || [ -z "${WQM_DEV_ROOT:-}" ]; then
    printf '%s' "$_p"
    return 0
  fi
  case "$_p" in
    "$WQM_HOST_DEV_ROOT"|"$WQM_HOST_DEV_ROOT"/*)
      printf '%s%s' "$WQM_DEV_ROOT" "${_p#"$WQM_HOST_DEV_ROOT"}"
      return 0
      ;;
  esac
  printf '%s' "$_p"
}

REPO_ROOT="$(_translate_to_container "$REPO_ROOT")"
if [ -n "$WORKTREE_PATH" ]; then
  WORKTREE_PATH="$(_translate_to_container "$WORKTREE_PATH")"
fi

PROJECT_NAME="$(wqm_path_basename "$REPO_ROOT")"

REPO_ROOT_E="$(wqm_path_json_escape "$REPO_ROOT")"
PROJECT_NAME_E="$(wqm_path_json_escape "$PROJECT_NAME")"
BRANCH_E="$(wqm_path_json_escape "$BRANCH")"
COMMIT_E="$(wqm_path_json_escape "$COMMIT")"
REMOTE_E="$(wqm_path_json_escape "$REMOTE")"
WORKTREE_PATH_E="$(wqm_path_json_escape "$WORKTREE_PATH")"
HOOK_NAME_E="$(wqm_path_json_escape "$HOOK_NAME")"

write_log() {
  [ -n "$LOG_FILE" ] || return 0
  _dir="$(dirname "$LOG_FILE")"
  [ -d "$_dir" ] || mkdir -p "$_dir" 2>/dev/null || return 0
  printf '%s hook=%s repo=%s branch=%s status=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$HOOK_NAME" "$REPO_ROOT" "$BRANCH" "$1" \
    >> "$LOG_FILE" 2>/dev/null || true
}

# Temp file for response headers (curl `-D <file>`). `/dev/stderr` is not
# portable to Git Bash on Windows ("Failed to open /proc/self/fd/2").
HEADERS_FILE="$(mktemp 2>/dev/null || printf '/tmp/wqm-hook-headers-%d' "$$")"
trap 'rm -f "$HEADERS_FILE"' EXIT INT TERM

# POST to MCP. Args: $1=payload, $2=session-id (may be empty).
# Body → stdout. Headers → $HEADERS_FILE (overwritten each call).
curl_post() {
  _payload="$1"
  _session="${2:-}"

  if [ -n "$_session" ] && [ -n "$MCP_TOKEN" ]; then
    curl --silent --show-error --max-time "$TIMEOUT" -D "$HEADERS_FILE" \
      -H "Authorization: Bearer $MCP_TOKEN" \
      -H "Mcp-Session-Id: $_session" \
      -H "Content-Type: application/json" \
      -H "Accept: application/json, text/event-stream" \
      --data "$_payload" "$MCP_URL"
  elif [ -n "$_session" ]; then
    curl --silent --show-error --max-time "$TIMEOUT" -D "$HEADERS_FILE" \
      -H "Mcp-Session-Id: $_session" \
      -H "Content-Type: application/json" \
      -H "Accept: application/json, text/event-stream" \
      --data "$_payload" "$MCP_URL"
  elif [ -n "$MCP_TOKEN" ]; then
    curl --silent --show-error --max-time "$TIMEOUT" -D "$HEADERS_FILE" \
      -H "Authorization: Bearer $MCP_TOKEN" \
      -H "Content-Type: application/json" \
      -H "Accept: application/json, text/event-stream" \
      --data "$_payload" "$MCP_URL"
  else
    curl --silent --show-error --max-time "$TIMEOUT" -D "$HEADERS_FILE" \
      -H "Content-Type: application/json" \
      -H "Accept: application/json, text/event-stream" \
      --data "$_payload" "$MCP_URL"
  fi
}

# ── Step 1: initialize MCP session ────────────────────────────────────────────
INIT_PAYLOAD='{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"wqm-git-hook","version":"0.1"}}}'

curl_post "$INIT_PAYLOAD" '' >/dev/null 2>&1 || true

# Extract Mcp-Session-Id from the headers file. Case-insensitive because
# Streamable HTTP servers may emit `Mcp-Session-Id` or `mcp-session-id`.
SESSION_ID="$(
  tr -d '\r' < "$HEADERS_FILE" 2>/dev/null \
    | awk 'BEGIN{IGNORECASE=1} /^Mcp-Session-Id:/ {sub(/^[^:]+:[ \t]*/, "", $0); print; exit}'
)"

if [ -z "$SESSION_ID" ]; then
  write_log init_failed
  exit 0
fi

# ── Step 2: notifications/initialized (acknowledge) ───────────────────────────
INITIALIZED_NOTIFY='{"jsonrpc":"2.0","method":"notifications/initialized"}'
curl_post "$INITIALIZED_NOTIFY" "$SESSION_ID" >/dev/null 2>&1 || true

# ── Step 3: tools/call → workspace_index.sync_current_branch ──────────────────
TOOL_CALL_PAYLOAD=$(printf '%s' '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"workspace_index","arguments":{"action":"sync_current_branch","repoDir":"'"$REPO_ROOT_E"'","projectName":"'"$PROJECT_NAME_E"'","currentBranch":"'"$BRANCH_E"'","commitHash":"'"$COMMIT_E"'","worktreePath":"'"$WORKTREE_PATH_E"'","isWorktree":'"$IS_WORKTREE"',"gitRemote":"'"$REMOTE_E"'","hookName":"'"$HOOK_NAME_E"'"}}}')

RESPONSE="$(curl_post "$TOOL_CALL_PAYLOAD" "$SESSION_ID" 2>/dev/null)" || RESPONSE=""

if [ -n "$LOG_FILE" ]; then
  _dir="$(dirname "$LOG_FILE")"
  [ -d "$_dir" ] || mkdir -p "$_dir" 2>/dev/null || true
  printf '%s hook=%s repo=%s branch=%s response=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$HOOK_NAME" "$REPO_ROOT" "$BRANCH" "$RESPONSE" \
    >> "$LOG_FILE" 2>/dev/null || true
fi

exit 0
