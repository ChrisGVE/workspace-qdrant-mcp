#!/usr/bin/env bash
# qdrant-quarantine-wrapper.sh — wraps Qdrant startup with auto-quarantine of
# corrupted collections.
#
# Background: a host reboot mid-write can leave a collection segment in an
# inconsistent state that triggers a Rust panic on load (e.g. gridstore.rs
# `LiteralOutOfBounds`). Qdrant's default behavior is to crash and the
# `restart: always` policy retries the same broken state forever, blocking
# every container that has `depends_on: qdrant healthy`.
#
# This wrapper runs the real Qdrant entrypoint, captures stderr, and if it
# fails with a known shard-load panic, moves the offending collection
# directory aside (`/qdrant/storage/collections/<name>` -> `.corrupted_<ts>_<name>`)
# and retries. The collection re-appears empty; memexd recreates it via
# `ensure_collection` on next startup and re-enqueues missing files via
# `ignore_sync`.
#
# Failure modes that DO NOT match (and propagate normally):
#   - Out of memory / OOMkill
#   - Config file errors
#   - Permission errors on volume
#   - Panics outside of shard loading
#
# Exit codes:
#   0   — Qdrant ran and exited cleanly (or restarted successfully after quarantine)
#   1   — failed and either no quarantine pattern matched, or max retries exceeded
#   2   — wrapper internal error (mv failed, missing entrypoint, etc.)

set -e

MAX_RETRIES="${QDRANT_QUARANTINE_MAX_RETRIES:-3}"
STORAGE="${QDRANT_STORAGE_DIR:-/qdrant/storage}"
WRAPPER_LOG_PREFIX="[qdrant-quarantine]"

# Locate Qdrant's original entrypoint. The official image ships
# `./entrypoint.sh` in WORKDIR `/qdrant`. Fall back to direct binary if the
# layout changes upstream.
real_entrypoint() {
  if [ -x "/qdrant/entrypoint.sh" ]; then
    echo "/qdrant/entrypoint.sh"
  elif [ -x "/qdrant/qdrant" ]; then
    echo "/qdrant/qdrant"
  else
    echo ""
  fi
}

ENTRYPOINT_PATH="$(real_entrypoint)"
if [ -z "$ENTRYPOINT_PATH" ]; then
  echo "$WRAPPER_LOG_PREFIX FATAL: cannot locate qdrant entrypoint or binary" >&2
  exit 2
fi
echo "$WRAPPER_LOG_PREFIX wrapping $ENTRYPOINT_PATH (max_retries=$MAX_RETRIES)" >&2

# Quarantine any collection mentioned in a "Failed to load local shard" line
# from $1 (log file path). Returns 0 if at least one collection was moved,
# 1 if no recognized pattern matched. Writes a marker line to
# $STORAGE/.quarantine_log with `<iso8601_utc>|<collection>|<reason>` so the
# daemon can later detect drift and trigger reembed.
quarantine_corrupted() {
  local log_path="$1"
  local moved=0
  local colls
  local ts
  local c
  local src
  local dst
  # Pattern: ERROR ... "Failed to load local shard "./storage/collections/<name>/..."
  # Extract <name> uniquely.
  colls=$(grep -oE 'Failed to load local shard "./storage/collections/[^/]+/' "$log_path" 2>/dev/null \
          | sed -E 's|.*/collections/([^/]+)/.*|\1|' \
          | sort -u)
  if [ -z "$colls" ]; then
    # Secondary pattern: gridstore LiteralOutOfBounds with collection in nearby line.
    # We can't reliably map without context, so just log and skip.
    if grep -q "gridstore.*LiteralOutOfBounds" "$log_path" 2>/dev/null; then
      echo "$WRAPPER_LOG_PREFIX gridstore panic detected but no collection name in 'Failed to load local shard' line; not quarantining blindly" >&2
    fi
    return 1
  fi

  ts=$(date -u +%Y%m%d_%H%M%SZ)
  for c in $colls; do
    src="$STORAGE/collections/$c"
    dst="$STORAGE/.corrupted_${ts}_${c}"
    if [ ! -d "$src" ]; then
      echo "$WRAPPER_LOG_PREFIX collection $c referenced in panic but dir $src not found, skipping" >&2
      continue
    fi
    if mv "$src" "$dst" 2>/dev/null; then
      echo "$WRAPPER_LOG_PREFIX QUARANTINED $c -> .corrupted_${ts}_${c}" >&2
      printf '%s|%s|shard-load-panic\n' "$ts" "$c" >> "$STORAGE/.quarantine_log" 2>/dev/null || true
      moved=$((moved + 1))
    else
      echo "$WRAPPER_LOG_PREFIX ERROR moving $c aside (perm/disk issue), propagating original failure" >&2
      return 1
    fi
  done

  if [ "$moved" -gt 0 ]; then
    return 0
  fi
  return 1
}

run_once() {
  # First arg is the log path (internal); the rest are qdrant's CLI args.
  local log_path="$1"
  shift
  set +e
  # Mirror qdrant's combined stdout+stderr to BOTH the container log (so
  # `docker logs wqm-qdrant` keeps working) and a file we can grep after the
  # process exits to drive quarantine decisions.
  # Capture qdrant's exit code (not tee's) via PIPESTATUS.
  "$ENTRYPOINT_PATH" "$@" 2>&1 | tee -- "$log_path"
  local rc=${PIPESTATUS[0]}
  set -e
  return "$rc"
}

retry=0
log_file=$(mktemp /tmp/qdrant-wrapper.XXXXXX.log)
trap 'rm -f "$log_file"' EXIT

while [ $retry -le $MAX_RETRIES ]; do
  : > "$log_file"  # truncate between attempts
  if run_once "$log_file" "$@"; then
    exit 0
  fi
  echo "$WRAPPER_LOG_PREFIX qdrant exited with failure on attempt $((retry + 1))" >&2

  if quarantine_corrupted "$log_file"; then
    retry=$((retry + 1))
    echo "$WRAPPER_LOG_PREFIX retrying ($retry/$MAX_RETRIES)..." >&2
    continue
  fi

  echo "$WRAPPER_LOG_PREFIX no quarantineable pattern matched, propagating exit failure" >&2
  exit 1
done

echo "$WRAPPER_LOG_PREFIX max retries ($MAX_RETRIES) exceeded; giving up" >&2
exit 1
