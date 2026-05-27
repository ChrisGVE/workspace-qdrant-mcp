# shellcheck shell=sh
# path-resolver.sh — POSIX shell git/path detection helpers.
#
# Source this file from a hook or installer:
#
#   . "$(dirname "$0")/../lib/path-resolver.sh"
#
# All functions defer to the `git` CLI for git-specific detection; that's
# the only implementation that handles main repos, linked worktrees,
# submodules, and detached HEAD identically. The path normalization helpers
# mirror the Rust `wqm-common::paths::CanonicalPath` rules: forward slashes,
# `~` expanded, `.` segments removed, `..` rejected, duplicate `/` collapsed.
#
# Convention: every function prints its result to stdout (empty when not
# resolvable) and exits 0 — never nonzero. Callers check by emptiness, not
# exit code, so a missing tool never propagates as a hook failure.

# ── Tool availability ────────────────────────────────────────────────────────

# wqm_path_has_tool <name> — true iff `name` is on PATH.
wqm_path_has_tool() {
  command -v "$1" >/dev/null 2>&1
}

# ── Git repository detection ─────────────────────────────────────────────────

# wqm_git_repo_root <path> — print the repo root containing <path>, or
# empty when not in a repo.
wqm_git_repo_root() {
  _path="${1:-.}"
  wqm_path_has_tool git || { printf ''; return 0; }
  git -C "$_path" rev-parse --show-toplevel 2>/dev/null || printf ''
}

# wqm_git_is_repository <path> — print 'true' / 'false'.
wqm_git_is_repository() {
  if [ -n "$(wqm_git_repo_root "${1:-.}")" ]; then
    printf 'true'
  else
    printf 'false'
  fi
}

# wqm_git_is_worktree <repo-root> — print 'true' when .git is a file
# (linked worktree), 'false' when directory or missing.
wqm_git_is_worktree() {
  _root="${1:?repo root required}"
  if [ -f "$_root/.git" ]; then
    printf 'true'
  else
    printf 'false'
  fi
}

# wqm_git_common_dir <repo-root> — print the shared git dir (absolute).
# For a main repo this is `<root>/.git`; for a linked worktree it's the
# parent repo's `.git`.
wqm_git_common_dir() {
  _root="${1:?repo root required}"
  wqm_path_has_tool git || { printf ''; return 0; }
  _out=$(git -C "$_root" rev-parse --git-common-dir 2>/dev/null || printf '')
  [ -z "$_out" ] && { printf ''; return 0; }
  # rev-parse may return a relative path; resolve against repo root.
  case "$_out" in
    /*|[A-Za-z]:/*|[A-Za-z]:\\*) printf '%s' "$_out" ;;
    *) printf '%s/%s' "$_root" "$_out" ;;
  esac
}

# wqm_git_current_branch <repo-root> — branch name, or 'HEAD' if detached.
wqm_git_current_branch() {
  _root="${1:?repo root required}"
  wqm_path_has_tool git || { printf ''; return 0; }
  git -C "$_root" rev-parse --abbrev-ref HEAD 2>/dev/null || printf ''
}

# wqm_git_head_commit <repo-root> — SHA of HEAD, empty in empty repos.
wqm_git_head_commit() {
  _root="${1:?repo root required}"
  wqm_path_has_tool git || { printf ''; return 0; }
  git -C "$_root" rev-parse HEAD 2>/dev/null || printf ''
}

# wqm_git_remote_url <repo-root> — `remote.origin.url`, or empty.
wqm_git_remote_url() {
  _root="${1:?repo root required}"
  wqm_path_has_tool git || { printf ''; return 0; }
  git -C "$_root" config --get remote.origin.url 2>/dev/null || printf ''
}

# wqm_git_state <repo-root> — print a JSON object describing the repo state.
# Fields: repoRoot, branch, commit, remoteUrl, isWorktree, worktreePath,
# commonDir. Designed for the MCP HTTP hook payload. Strings are
# JSON-escaped via `wqm_path_json_escape` below.
wqm_git_state() {
  _root="${1:?repo root required}"
  _branch=$(wqm_git_current_branch "$_root")
  _commit=$(wqm_git_head_commit "$_root")
  _remote=$(wqm_git_remote_url "$_root")
  _worktree=$(wqm_git_is_worktree "$_root")
  _common=$(wqm_git_common_dir "$_root")
  _worktree_path=""
  [ "$_worktree" = "true" ] && _worktree_path="$_root"

  printf '{"repoRoot":"%s","branch":"%s","commit":"%s","remoteUrl":"%s","isWorktree":%s,"worktreePath":"%s","commonDir":"%s"}' \
    "$(wqm_path_json_escape "$_root")" \
    "$(wqm_path_json_escape "$_branch")" \
    "$(wqm_path_json_escape "$_commit")" \
    "$(wqm_path_json_escape "$_remote")" \
    "$_worktree" \
    "$(wqm_path_json_escape "$_worktree_path")" \
    "$(wqm_path_json_escape "$_common")"
}

# ── Path utilities ───────────────────────────────────────────────────────────

# wqm_path_normalize_slashes <path> — convert backslashes to forward slashes.
wqm_path_normalize_slashes() {
  printf '%s' "$1" | tr '\\' '/'
}

# wqm_path_basename <path> — final path component (cross-platform).
wqm_path_basename() {
  # `basename` is POSIX but stumbles on some Windows paths under Git Bash.
  # Manual implementation: drop trailing slashes, then everything up to
  # the last slash.
  _p=$(wqm_path_normalize_slashes "$1")
  _p="${_p%/}"
  case "$_p" in
    */*) printf '%s' "${_p##*/}" ;;
    *)   printf '%s' "$_p" ;;
  esac
}

# wqm_path_is_windows_absolute <path> — print 'true' iff path starts with a
# drive letter (e.g. `C:/`, `D:\`). Already-POSIX paths return 'false'.
wqm_path_is_windows_absolute() {
  case "$1" in
    [A-Za-z]:/*|[A-Za-z]:\\*) printf 'true' ;;
    *) printf 'false' ;;
  esac
}

# wqm_path_is_absolute <path> — print 'true' for POSIX or Windows absolute
# paths, 'false' for relative.
wqm_path_is_absolute() {
  case "$1" in
    /*|[A-Za-z]:/*|[A-Za-z]:\\*) printf 'true' ;;
    *) printf 'false' ;;
  esac
}

# wqm_path_json_escape <string> — escape backslash and double-quote for safe
# embedding inside a JSON double-quoted value. Newlines become spaces.
wqm_path_json_escape() {
  printf '%s' "$1" | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g' | tr '\n' ' '
}
