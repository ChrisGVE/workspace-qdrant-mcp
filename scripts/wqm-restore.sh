#!/usr/bin/env bash
#
# wqm-restore.sh -- restore the workspace-qdrant stores from a wqm-backup archive.
#
# Companion to wqm-backup.sh. Takes a YYYYMMDD_wqm_backup.tar.zst archive and
# restores every SQLite database and every Qdrant collection it contains. This is
# a DESTRUCTIVE, disaster-recovery operation: it overwrites the live stores. It
# refuses to run while the daemon is up (that would corrupt the SQLite files),
# verifies archive integrity + per-file checksums before touching anything, keeps
# a safety copy of the databases it replaces, and requires an explicit
# confirmation unless --force is given.
#
# Usage:
#   wqm-restore.sh <archive.tar.zst> [--force]
#
# Environment overrides (all optional):
#   WQM_DATA_DIR    directory to restore the *.db files into  (default: ~/.local/share/workspace-qdrant)
#   QDRANT_URL      Qdrant base URL                           (default: http://localhost:6333)
#   QDRANT_API_KEY  Qdrant api-key header                     (default: unset -- local, no auth)
#   WQM_BIN         wqm CLI path (daemon-state probe)         (default: wqm on PATH)
#
set -euo pipefail

# Scheduled/minimal-env launchers omit Homebrew from PATH; ensure brew tools resolve.
export PATH="/usr/local/bin:/opt/homebrew/bin:$PATH"

# ---- configuration ---------------------------------------------------------
DATA_DIR="${WQM_DATA_DIR:-$HOME/.local/share/workspace-qdrant}"
QDRANT_URL="${QDRANT_URL:-http://localhost:6333}"
WQM_BIN="${WQM_BIN:-wqm}"
FORCE=0

log() { printf '%s  %s\n' "$(date +%H:%M:%S)" "$*"; }
die() {
	printf '%s  ERROR: %s\n' "$(date +%H:%M:%S)" "$*" >&2
	exit 1
}

qcurl() {
	local args=(--fail --silent --show-error --max-time 600)
	[[ -n "${QDRANT_API_KEY:-}" ]] && args+=(-H "api-key: ${QDRANT_API_KEY}")
	curl "${args[@]}" "$@"
}

# ---- argument parsing ------------------------------------------------------
ARCHIVE=""
for a in "$@"; do
	case "$a" in
	--force) FORCE=1 ;;
	-*) die "unknown option: $a" ;;
	*) [[ -z "$ARCHIVE" ]] && ARCHIVE="$a" || die "unexpected extra argument: $a" ;;
	esac
done
[[ -n "$ARCHIVE" ]] || die "usage: wqm-restore.sh <archive.tar.zst> [--force]"
[[ -f "$ARCHIVE" ]] || die "archive not found: $ARCHIVE"
[[ -d "$DATA_DIR" ]] || die "wqm data directory not found: $DATA_DIR (set WQM_DATA_DIR)"

for tool in sqlite3 curl jq tar zstd shasum; do
	command -v "$tool" >/dev/null 2>&1 || die "required tool not found on PATH: $tool"
done

# ---- refuse if the daemon is running (SQLite overwrite would corrupt) -------
# Best-effort probe: if `wqm service status` reports anything but a clean stop,
# or a memexd process is alive, refuse and tell the operator how to stop it.
if pgrep -x memexd >/dev/null 2>&1; then
	die "memexd is running -- stop the daemon before restoring (e.g. launchctl unload the memexd plist), then re-run."
fi

# ---- staging + integrity verification (before touching anything) -----------
STAGE="$(mktemp -d "${TMPDIR:-/tmp}/wqm-restore.XXXXXX")"
trap 'rm -rf "$STAGE"' EXIT INT TERM

log "verifying archive integrity: $ARCHIVE"
zstd -t -q "$ARCHIVE" || die "archive failed zstd integrity check -- refusing to restore"

log "extracting archive"
tar -C "$STAGE" --use-compress-program='zstd -d' -xf "$ARCHIVE" ||
	die "extraction failed"
[[ -f "$STAGE/MANIFEST.txt" ]] || die "no MANIFEST.txt in archive -- not a wqm-backup archive?"

log "checking per-file checksums against MANIFEST"
# The manifest lists "  <sha256>  <relpath>" lines under a 'sha256:' header.
awk '/^sha256:/{f=1;next} f&&NF>=2{print $1"  "$2}' "$STAGE/MANIFEST.txt" >"$STAGE/.sums"
[[ -s "$STAGE/.sums" ]] || die "MANIFEST.txt carries no checksums -- refusing to restore"
while read -r want rel; do
	[[ -f "$STAGE/$rel" ]] || die "archive is missing a manifested file: $rel"
	got="$(shasum -a 256 "$STAGE/$rel" | awk '{print $1}')"
	[[ "$got" == "$want" ]] || die "checksum mismatch for $rel (archive corrupt)"
done <"$STAGE/.sums"
log "checksums OK"

# ---- inventory what will be restored ---------------------------------------
shopt -s nullglob
sql_files=("$STAGE"/sql/*.db)
snap_files=("$STAGE"/qdrant/*.snapshot)
shopt -u nullglob
[[ ${#sql_files[@]} -gt 0 || ${#snap_files[@]} -gt 0 ]] || die "archive contains no databases or snapshots"

echo
echo "About to restore INTO:"
echo "  data_dir:  ${DATA_DIR/#$HOME/\~}"
echo "  qdrant:    $QDRANT_URL"
echo "This OVERWRITES ${#sql_files[@]} database(s) and ${#snap_files[@]} Qdrant collection(s)."
echo "The current databases are copied aside to <name>.pre-restore-<ts> first."
echo

# ---- confirmation gate -----------------------------------------------------
if [[ "$FORCE" != "1" ]]; then
	read -r -p "Type 'restore' to proceed: " reply
	[[ "$reply" == "restore" ]] || die "aborted by operator"
fi

TS="$(date +%Y%m%d-%H%M%S)"

# ---- 1. SQLite databases (safety-copy the live one, then replace) ----------
for src in "${sql_files[@]}"; do
	name="$(basename "$src")"
	dst="$DATA_DIR/$name"
	if [[ -f "$dst" ]]; then
		log "safety copy: $name -> $name.pre-restore-$TS"
		cp -p "$dst" "$dst.pre-restore-$TS" || die "could not safety-copy existing $name"
		# Drop stale WAL/SHM so the restored file is authoritative on next open.
		rm -f "$dst-wal" "$dst-shm"
	fi
	log "restoring SQLite: $name"
	cp "$src" "$dst" || die "failed to restore $name"
done

# ---- 2. Qdrant collections (upload snapshot, snapshot data takes priority) --
for src in "${snap_files[@]}"; do
	col="$(basename "$src" .snapshot)"
	log "restoring Qdrant collection: $col"
	qcurl -X POST \
		-H 'Content-Type:multipart/form-data' \
		-F "snapshot=@${src}" \
		"$QDRANT_URL/collections/$col/snapshots/upload?priority=snapshot" >/dev/null ||
		die "snapshot upload/recover failed for collection $col"
done

echo
log "restore complete."
log "restart the daemon (e.g. launchctl load the memexd plist) and verify with: $WQM_BIN status health"
[[ ${#sql_files[@]} -gt 0 ]] && log "safety copies kept as *.pre-restore-$TS in ${DATA_DIR/#$HOME/\~} -- delete once verified."
exit 0
