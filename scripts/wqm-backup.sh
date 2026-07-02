#!/usr/bin/env bash
#
# wqm-backup.sh -- point-in-time backup of the workspace-qdrant stores.
#
# Produces ONE compressed archive: every Qdrant collection snapshot + every
# SQLite database, named  YYYYMMDD_wqm_backup.tar.zst,  written to the directory
# given as the single argument. Intended to be triggered unattended (e.g. a
# Keyboard Maestro macro at 03:00) -- non-interactive, self-contained, exits 0
# only on a fully verified archive.
#
# This is the INTERIM backup: it relies on each store's own consistent-snapshot
# mechanism (SQLite online .backup API; Qdrant snapshot API), which are each
# internally consistent while the daemon runs. It does not yet use a global
# daemon quiesce hook (that is a wqm-0.2 Phase-2 deliverable, DI-2); a best-effort
# watcher pause narrows the cross-store window but is not required for a usable
# backup.
#
# Usage:
#   wqm-backup.sh <destination-directory>
#
# Environment overrides (all optional):
#   WQM_DATA_DIR      directory holding the *.db files   (default: ~/.local/share/workspace-qdrant)
#   QDRANT_URL        Qdrant base URL                     (default: http://localhost:6333)
#   QDRANT_API_KEY    Qdrant api-key header               (default: unset -- local, no auth)
#   ZSTD_LEVEL        zstd compression level              (default: 10)
#   WQM_PAUSE_WATCHER 1 to best-effort pause the watcher  (default: 0 -- off; daemon may be degraded)
#   WQM_BIN           wqm CLI path (only used if pausing)  (default: wqm on PATH)
#
set -euo pipefail

# Scheduled launchers (Keyboard Maestro, launchd, cron) run with a minimal PATH
# that omits Homebrew; make sure brew-installed tools (zstd/jq/curl) resolve.
export PATH="/usr/local/bin:/opt/homebrew/bin:$PATH"

# ---- configuration ---------------------------------------------------------
DATA_DIR="${WQM_DATA_DIR:-$HOME/.local/share/workspace-qdrant}"
QDRANT_URL="${QDRANT_URL:-http://localhost:6333}"
ZSTD_LEVEL="${ZSTD_LEVEL:-10}"
PAUSE_WATCHER="${WQM_PAUSE_WATCHER:-0}"
WQM_BIN="${WQM_BIN:-wqm}"

log() { printf '%s  %s\n' "$(date +%H:%M:%S)" "$*"; }
die() {
	printf '%s  ERROR: %s\n' "$(date +%H:%M:%S)" "$*" >&2
	exit 1
}

# curl with optional api-key; --fail so HTTP errors abort the step.
qcurl() {
	local args=(--fail --silent --show-error --max-time 600)
	[[ -n "${QDRANT_API_KEY:-}" ]] && args+=(-H "api-key: ${QDRANT_API_KEY}")
	curl "${args[@]}" "$@"
}

# ---- argument + environment checks -----------------------------------------
[[ $# -eq 1 ]] || die "exactly one argument required: <destination-directory> (got $#)"
DEST_DIR="$1"
[[ -d "$DEST_DIR" ]] || die "destination directory does not exist: $DEST_DIR"
[[ -w "$DEST_DIR" ]] || die "destination directory is not writable: $DEST_DIR"
[[ -d "$DATA_DIR" ]] || die "wqm data directory not found: $DATA_DIR (set WQM_DATA_DIR)"

for tool in sqlite3 curl jq tar zstd; do
	command -v "$tool" >/dev/null 2>&1 || die "required tool not found on PATH: $tool"
done

DATESTAMP="$(date +%Y%m%d)"
BACKUP_NAME="${DATESTAMP}_wqm_backup"
ARCHIVE_PATH="${DEST_DIR%/}/${BACKUP_NAME}.tar.zst"

# Staging dir (auto-removed) + guaranteed watcher-resume, whatever happens.
STAGE="$(mktemp -d "${TMPDIR:-/tmp}/wqm-backup.XXXXXX")"
WATCHER_PAUSED=0
cleanup() {
	local rc=$?
	if [[ "$WATCHER_PAUSED" == "1" ]]; then
		log "resuming file watcher"
		if ! timeout 20 "$WQM_BIN" project watch resume >/dev/null 2>&1; then
			printf '%s  WARNING: could not resume the watcher -- run `%s project watch resume` manually\n' \
				"$(date +%H:%M:%S)" "$WQM_BIN" >&2
		fi
	fi
	rm -rf "$STAGE"
	exit "$rc"
}
trap cleanup EXIT INT TERM

mkdir -p "$STAGE/sql" "$STAGE/qdrant"
log "wqm backup starting -> $ARCHIVE_PATH"
log "data dir: $DATA_DIR   qdrant: $QDRANT_URL"

# ---- optional best-effort watcher pause ------------------------------------
if [[ "$PAUSE_WATCHER" == "1" ]]; then
	log "pausing file watcher (best-effort)"
	if timeout 20 "$WQM_BIN" project watch pause >/dev/null 2>&1; then
		WATCHER_PAUSED=1
	else
		log "watcher pause failed or timed out -- continuing (per-store snapshots stay consistent)"
	fi
fi

# ---- 1. SQLite databases (online .backup -- safe while the daemon writes) ----
shopt -s nullglob
db_files=("$DATA_DIR"/*.db)
shopt -u nullglob
[[ ${#db_files[@]} -gt 0 ]] || die "no *.db files found in $DATA_DIR"

for db in "${db_files[@]}"; do
	name="$(basename "$db")"
	log "backing up SQLite: $name"
	# .timeout waits out a busy lock; .backup uses SQLite's consistent online backup.
	sqlite3 "$db" ".timeout 60000" ".backup '$STAGE/sql/$name'" ||
		die "sqlite .backup failed for $name"
done

# ---- 2. Qdrant per-collection snapshots ------------------------------------
mapfile -t collections < <(qcurl "$QDRANT_URL/collections" | jq -r '.result.collections[].name' | sort)
[[ ${#collections[@]} -gt 0 ]] || die "no Qdrant collections returned by $QDRANT_URL"

for col in "${collections[@]}"; do
	log "snapshotting Qdrant collection: $col"
	snap="$(qcurl -X POST "$QDRANT_URL/collections/$col/snapshots?wait=true" | jq -r '.result.name')"
	[[ -n "$snap" && "$snap" != "null" ]] || die "snapshot creation returned no name for collection $col"
	qcurl -o "$STAGE/qdrant/$col.snapshot" "$QDRANT_URL/collections/$col/snapshots/$snap" ||
		die "snapshot download failed for collection $col"
	# Remove the server-side snapshot so Qdrant's snapshot dir does not accumulate.
	qcurl -X DELETE "$QDRANT_URL/collections/$col/snapshots/$snap" >/dev/null ||
		log "WARNING: could not delete server-side snapshot $snap for $col (harmless, but it lingers)"
done

# ---- 3. manifest (provenance + integrity) ----------------------------------
{
	echo "wqm backup manifest"
	echo "created:       $(date -u +%Y-%m-%dT%H:%M:%SZ) (UTC)"
	# Home-relativized + no hostname: the manifest must not carry the operator's
	# absolute paths or machine name inside a backup that may travel off-machine.
	echo "data_dir:      ${DATA_DIR/#$HOME/\~}"
	echo "qdrant_url:    $QDRANT_URL"
	echo "qdrant_version: $(qcurl "$QDRANT_URL/" | jq -r '.version' 2>/dev/null || echo unknown)"
	echo "collections:   ${collections[*]}"
	echo "databases:     $(printf '%s ' "${db_files[@]##*/}")"
	echo ""
	echo "sha256:"
	(cd "$STAGE" && find sql qdrant -type f | sort | while read -r f; do
		echo "  $(shasum -a 256 "$f" | awk '{print $1}')  $f"
	done)
} >"$STAGE/MANIFEST.txt"

# ---- 4. compress -> single archive -----------------------------------------
log "compressing archive (zstd -${ZSTD_LEVEL}, multithreaded)"
tar -C "$STAGE" -cf - MANIFEST.txt sql qdrant |
	zstd -T0 "-${ZSTD_LEVEL}" -q -o "$ARCHIVE_PATH" -f ||
	die "compression failed"

# ---- 5. verify the archive is readable + complete --------------------------
# zstd -t decompresses and checksums the whole archive -- the real integrity gate.
# (A `tar -t` re-listing is intentionally NOT used: tar stops at the archive-end
# marker before zstd flushes its trailing frame bytes, which emits a harmless but
# alarming "Broken pipe" on stderr -- noise no unattended job should print.)
log "verifying archive"
zstd -t -q "$ARCHIVE_PATH" || die "archive failed zstd integrity check"
n_members=$((1 + ${#db_files[@]} + ${#collections[@]})) # MANIFEST + dbs + snapshots
size="$(du -h "$ARCHIVE_PATH" | awk '{print $1}')"
log "OK -- $ARCHIVE_PATH ($size; $n_members members: MANIFEST + ${#db_files[@]} db + ${#collections[@]} snapshots)"
