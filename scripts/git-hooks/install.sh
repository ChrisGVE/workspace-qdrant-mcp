#!/bin/sh
# install.sh — install workspace-qdrant MCP git hooks into a repo.
#
# Companion to scripts/windows/indexed-projects-hooks.ps1, but for the
# dockerized MCP HTTP setup. Installs five hooks (post-checkout, post-commit,
# post-merge, post-rewrite, post-worktree-add) that POST to the MCP HTTP
# endpoint so each git event refreshes the registered project/branch/worktree
# in the daemon.
#
# Usage:
#   scripts/git-hooks/install.sh [--repo <path>] [--hooks-dir <path>]
#                                [--mcp-url <url>] [--token <bearer>]
#                                [--log <path>] [--uninstall]
#
# Defaults:
#   --repo       = $(git rev-parse --show-toplevel)
#   --hooks-dir  = <git common dir>/hooks   (works for worktrees too)
#   --mcp-url    = http://localhost:6335/mcp
#   --token      = $MCP_HTTP_TOKEN (env)
#   --log        = <repo>/.wqm-fork/logs/git-hooks.jsonl
#
# After install, hooks call wqm-sync-branch.sh located at:
#   <wqm checkout>/scripts/git-hooks/wqm-sync-branch.sh
# Either pass --wqm-script explicitly or let the installer auto-detect
# (uses the directory containing install.sh).

set -eu

# ── Defaults ──────────────────────────────────────────────────────────────────
SELF_DIR="$(cd "$(dirname "$0")" && pwd)"
WQM_SCRIPT="$SELF_DIR/wqm-sync-branch.sh"

REPO=""
HOOKS_DIR=""
MCP_URL="${WQM_MCP_URL:-http://localhost:6335/mcp}"
TOKEN="${MCP_HTTP_TOKEN:-}"
LOG_FILE=""
HOST_DEV_ROOT="${WQM_HOST_DEV_ROOT:-}"
CONTAINER_DEV_ROOT="${WQM_DEV_ROOT:-}"
ENV_FILE=""
UNINSTALL=0
FORCE=0
# Track whether --hooks-dir was passed explicitly. When it was, we also
# set core.hooksPath so git actually looks there; when it defaulted to
# the git common hooks dir, no config change is needed.
EXPLICIT_HOOKS_DIR=0

# ── Arg parsing ───────────────────────────────────────────────────────────────
while [ $# -gt 0 ]; do
  case "$1" in
    --repo)        REPO="$2"; shift 2 ;;
    --hooks-dir)   HOOKS_DIR="$2"; EXPLICIT_HOOKS_DIR=1; shift 2 ;;
    --mcp-url)     MCP_URL="$2"; shift 2 ;;
    --token)       TOKEN="$2"; shift 2 ;;
    --log)         LOG_FILE="$2"; shift 2 ;;
    --wqm-script)  WQM_SCRIPT="$2"; shift 2 ;;
    --host-dev-root)      HOST_DEV_ROOT="$2"; shift 2 ;;
    --container-dev-root) CONTAINER_DEV_ROOT="$2"; shift 2 ;;
    --env-file)    ENV_FILE="$2"; shift 2 ;;
    --force)       FORCE=1; shift ;;
    --uninstall)   UNINSTALL=1; shift ;;
    -h|--help)
      sed -n '2,30p' "$0"
      exit 0
      ;;
    *)
      printf 'Unknown argument: %s\n' "$1" >&2
      exit 2
      ;;
  esac
done

# ── Resolve repo and hooks dir ────────────────────────────────────────────────
if [ -z "$REPO" ]; then
  REPO="$(git rev-parse --show-toplevel 2>/dev/null || true)"
fi
[ -n "$REPO" ] || { printf 'Not in a git repo; pass --repo <path>\n' >&2; exit 1; }
[ -d "$REPO" ] || { printf 'Repo path not found: %s\n' "$REPO" >&2; exit 1; }

if [ -z "$HOOKS_DIR" ]; then
  # git --git-common-dir returns the shared .git dir, which works for both
  # main repos and linked worktrees.
  GIT_COMMON_DIR="$(git -C "$REPO" rev-parse --git-common-dir 2>/dev/null || true)"
  [ -n "$GIT_COMMON_DIR" ] || { printf 'Cannot resolve git common dir\n' >&2; exit 1; }
  # Make absolute (git may return a relative path).
  case "$GIT_COMMON_DIR" in
    /*) ;;
    *) GIT_COMMON_DIR="$REPO/$GIT_COMMON_DIR" ;;
  esac
  HOOKS_DIR="$GIT_COMMON_DIR/hooks"
fi

[ -d "$HOOKS_DIR" ] || mkdir -p "$HOOKS_DIR"

# ── Default log path ──────────────────────────────────────────────────────────
if [ -z "$LOG_FILE" ]; then
  LOG_FILE="$REPO/.wqm-fork/logs/git-hooks.jsonl"
fi

# ── Auto-source docker/.env for missing values ────────────────────────────────
# When the daemon runs in Docker, the hook needs the container path so it can
# translate host paths before calling RegisterProject. We pick up WQM_DEV_ROOT,
# WQM_HOST_DEV_ROOT, and MCP_HTTP_TOKEN from docker/.env if not already set.
if [ -z "$ENV_FILE" ] && [ -f "$REPO/docker/.env" ]; then
  ENV_FILE="$REPO/docker/.env"
fi
if [ -n "$ENV_FILE" ] && [ -f "$ENV_FILE" ]; then
  while IFS= read -r _line || [ -n "$_line" ]; do
    case "$_line" in
      ''|'#'*) continue ;;
    esac
    _key="${_line%%=*}"
    _val="${_line#*=}"
    # Strip surrounding single/double quotes.
    case "$_val" in
      \"*\") _val="${_val#\"}"; _val="${_val%\"}" ;;
      \'*\') _val="${_val#\'}"; _val="${_val%\'}" ;;
    esac
    case "$_key" in
      WQM_DEV_ROOT)      [ -z "$CONTAINER_DEV_ROOT" ] && CONTAINER_DEV_ROOT="$_val" ;;
      WQM_HOST_DEV_ROOT) [ -z "$HOST_DEV_ROOT" ] && HOST_DEV_ROOT="$_val" ;;
      MCP_HTTP_TOKEN)    [ -z "$TOKEN" ] && TOKEN="$_val" ;;
    esac
  done < "$ENV_FILE"
fi

HOOK_LIST="post-checkout post-commit post-merge post-rewrite post-worktree-add"
MARKER="# WQM_SYNC_BRANCH_HOOK"

# ── Path translation helper ───────────────────────────────────────────────────
# When install.sh runs inside a container but the hooks must execute on the
# host (the common case for the dockerized MCP), translate paths from their
# container-visible form into the host-visible form before embedding them in
# the generated hook scripts or writing them to git config. Requires both
# --host-dev-root (path the host shell sees) and --container-dev-root (path
# bind-mounted inside the container) — without them we echo the input
# unchanged.
_translate_to_host() {
  _path="$1"
  if [ -n "$HOST_DEV_ROOT" ] && [ -n "$CONTAINER_DEV_ROOT" ]; then
    case "$_path" in
      "$CONTAINER_DEV_ROOT"|"$CONTAINER_DEV_ROOT"/*)
        printf '%s%s' "$HOST_DEV_ROOT" "${_path#"$CONTAINER_DEV_ROOT"}"
        return 0
        ;;
    esac
  fi
  printf '%s' "$_path"
}

# core.hooksPath, when we set it below, is read by host git, so it needs
# to be the host-visible path. HOOKS_DIR itself stays container-side
# because install.sh writes the files from inside the container.
HOOKS_DIR_HOST="$(_translate_to_host "$HOOKS_DIR")"

# ── Uninstall mode ────────────────────────────────────────────────────────────
if [ "$UNINSTALL" = "1" ]; then
  for h in $HOOK_LIST; do
    target="$HOOKS_DIR/$h"
    if [ -f "$target" ] && grep -q "$MARKER" "$target" 2>/dev/null; then
      rm -f "$target"
      printf 'removed: %s\n' "$target"
    fi
  done
  # Unset core.hooksPath only if it points at the dir we just cleaned out
  # (either the container-side path or the host-translated form, depending
  # on which was active when install last ran).
  _current="$(git -C "$REPO" config --get core.hooksPath 2>/dev/null || true)"
  if [ -n "$_current" ] && { [ "$_current" = "$HOOKS_DIR" ] || [ "$_current" = "$HOOKS_DIR_HOST" ]; }; then
    if git -C "$REPO" config --unset core.hooksPath 2>/dev/null; then
      printf 'unset: core.hooksPath\n'
    fi
  fi
  exit 0
fi

# ── Validate prerequisites ────────────────────────────────────────────────────
[ -f "$WQM_SCRIPT" ] || {
  printf 'wqm-sync-branch.sh not found at: %s\n' "$WQM_SCRIPT" >&2
  printf 'Pass --wqm-script <path> to override.\n' >&2
  exit 1
}
chmod +x "$WQM_SCRIPT" 2>/dev/null || true  # bind-mounted hosts may refuse chmod

WQM_SCRIPT="$(_translate_to_host "$WQM_SCRIPT")"
# WQM_HOOK_LOG inside each hook is opened by wqm-sync-branch.sh running on
# the host, so the embedded path also needs the host prefix. Embedded value
# only — install.sh itself never writes to LOG_FILE.
LOG_FILE_EMBED="$(_translate_to_host "$LOG_FILE")"

# Refuse to install if WQM_SCRIPT still lives under a container-only mount
# root (`/run/desktop/...` from Docker Desktop's WSL2 host bridge, or
# `/mnt/wsl/...`). Git hooks execute wherever `git` is invoked (the host
# shell, not a container), so a wrapper that points at a container-only
# path silently no-ops on the host. The wrapper itself runs (it's just a
# shell script) but the `exec` to WQM_SCRIPT fails with "command not
# found" and `|| true` swallows the error.
# Override with WQM_INSTALL_FORCE_CONTAINER_PATHS=1 if you really know.
case "$WQM_SCRIPT" in
  /run/desktop/*|/mnt/wsl/*)
    if [ -z "${WQM_INSTALL_FORCE_CONTAINER_PATHS:-}" ]; then
      printf 'ERROR: install.sh appears to be running inside a container or WSL bridge.\n' >&2
      printf '       WQM_SCRIPT resolved to: %s\n' "$WQM_SCRIPT" >&2
      printf '       which is not visible to git hooks running on the host.\n' >&2
      if [ -z "$HOST_DEV_ROOT" ] || [ -z "$CONTAINER_DEV_ROOT" ]; then
        printf '\n' >&2
        printf '       Fix: set WQM_HOST_DEV_ROOT (host path) and WQM_DEV_ROOT\n' >&2
        printf '       (container path) in your .env, then restart the MCP container.\n' >&2
        printf '       Or pass --host-dev-root <path> --container-dev-root <path>.\n' >&2
        printf '       Or re-run from the host shell (Git Bash on Windows, native sh on macOS/Linux).\n' >&2
      else
        printf '       host-dev-root=%s container-dev-root=%s did not match WQM_SCRIPT prefix.\n' \
          "$HOST_DEV_ROOT" "$CONTAINER_DEV_ROOT" >&2
      fi
      printf '       Override with WQM_INSTALL_FORCE_CONTAINER_PATHS=1 if you really know what you are doing.\n' >&2
      exit 1
    fi
    ;;
esac

# ── Generate each hook script ─────────────────────────────────────────────────
write_hook() {
  _hook_name="$1"
  _target="$HOOKS_DIR/$_hook_name"

  # post-checkout receives 3 args: prev_HEAD new_HEAD branch_checkout_flag
  # The flag is "0" for file checkouts (no branch change) — we skip those.
  _checkout_guard=""
  if [ "$_hook_name" = "post-checkout" ]; then
    _checkout_guard='if [ "${3:-}" = "0" ]; then exit 0; fi'
  fi

  cat > "$_target" <<EOF
#!/bin/sh
$MARKER
# Auto-generated by scripts/git-hooks/install.sh on $(date -u +%Y-%m-%dT%H:%M:%SZ)
# Do not edit; re-run install.sh to update.
$_checkout_guard
WQM_HOOK_NAME="$_hook_name" \\
WQM_MCP_URL="$MCP_URL" \\
WQM_MCP_TOKEN="$TOKEN" \\
WQM_HOOK_LOG="$LOG_FILE_EMBED" \\
WQM_HOST_DEV_ROOT="$HOST_DEV_ROOT" \\
WQM_DEV_ROOT="$CONTAINER_DEV_ROOT" \\
"$WQM_SCRIPT" "$_hook_name" >/dev/null 2>&1 || true
exit 0
EOF
  chmod +x "$_target" 2>/dev/null || true  # bind-mounted hosts may refuse chmod; files are usually +x already
  printf 'installed: %s\n' "$_target"
}

for h in $HOOK_LIST; do
  if [ -f "$HOOKS_DIR/$h" ] && ! grep -q "$MARKER" "$HOOKS_DIR/$h" 2>/dev/null; then
    if [ "$FORCE" = "1" ]; then
      printf 'overwriting non-wqm hook (--force): %s\n' "$HOOKS_DIR/$h" >&2
    else
      printf 'skipped (existing non-wqm hook, use --force to overwrite): %s\n' "$HOOKS_DIR/$h" >&2
      continue
    fi
  fi
  write_hook "$h"
done

# Clean up companion PS-era artifacts when forcing a switch to POSIX hooks.
if [ "$FORCE" = "1" ] && [ -f "$HOOKS_DIR/wqm-git-hook.ps1" ]; then
  rm -f "$HOOKS_DIR/wqm-git-hook.ps1"
  printf 'removed stale PS hook companion: %s\n' "$HOOKS_DIR/wqm-git-hook.ps1" >&2
fi

# Point git at the custom hooks dir. Skipped when HOOKS_DIR defaulted to
# the git common hooks dir (git looks there natively, so a config change
# would be a no-op at best and a confusing override at worst).
# Uses HOOKS_DIR_HOST so host-side git resolves the same files install.sh
# wrote via the container-side bind mount.
if [ "$EXPLICIT_HOOKS_DIR" = "1" ]; then
  if git -C "$REPO" config core.hooksPath "$HOOKS_DIR_HOST" 2>/dev/null; then
    printf 'configured: core.hooksPath = %s\n' "$HOOKS_DIR_HOST"
  else
    printf 'warning: failed to set core.hooksPath; hooks will not fire until you run:\n' >&2
    printf '         git -C %s config core.hooksPath %s\n' "$REPO" "$HOOKS_DIR_HOST" >&2
  fi
fi

printf '\nHooks installed in: %s\n' "$HOOKS_DIR"
printf 'MCP endpoint     : %s\n' "$MCP_URL"
printf 'Log file         : %s\n' "$LOG_FILE_EMBED"
if [ -n "$HOST_DEV_ROOT" ] && [ -n "$CONTAINER_DEV_ROOT" ]; then
  printf 'Path translation : %s -> %s\n' "$HOST_DEV_ROOT" "$CONTAINER_DEV_ROOT"
fi
if [ -z "$TOKEN" ]; then
  printf '\nWarning: MCP_HTTP_TOKEN is empty. Hooks will hit the MCP endpoint\n'
  printf '         unauthenticated and the request will be rejected.\n'
  printf '         Pass --token <value> or export MCP_HTTP_TOKEN before re-running.\n'
fi
if [ -n "$CONTAINER_DEV_ROOT" ] && [ -z "$HOST_DEV_ROOT" ]; then
  printf '\nWarning: WQM_DEV_ROOT is set but WQM_HOST_DEV_ROOT is empty. Path\n'
  printf '         translation is disabled, so host paths will be sent as-is.\n'
  printf '         The daemon will reject them if it cannot see the host path.\n'
fi
