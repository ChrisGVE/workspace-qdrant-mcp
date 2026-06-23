#!/usr/bin/env bash
#
# storage-guards.sh — branch-storage read/write boundary CI guards (arch §9, F1).
#
# Guard 1 (AC-F1.4): wqm-storage-write must NOT appear in mcp-server's feature
#   closure — the read-only consumer cannot pull the write crate transitively.
# Guard 3 (AC-F1.7): the mcp-server and wqm-cli release binaries must contain NO
#   Qdrant-mutating symbol (upsert_points / delete_points / overwrite_payload /
#   set_payload / create_collection). Those live ONLY in wqm-storage-write.
#
# Guard 2 (the trybuild compile-fail test) runs as an ordinary `cargo test`
# (storage/tests/guard2_read_cannot_write.rs), so it is not duplicated here.
#
# Run from the repo root:  bash scripts/ci/storage-guards.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT/src/rust"

MUTATING='upsert_points|delete_points|overwrite_payload|set_payload|create_collection'
fail=0

echo "== Guard 1: wqm-storage-write absent from mcp-server feature closure =="
if cargo tree -e features -p mcp-server | grep -q 'wqm-storage-write'; then
	echo "::error::Guard 1 FAILED — mcp-server's dependency closure reaches wqm-storage-write."
	cargo tree -e features -p mcp-server | grep -n 'wqm-storage-write' || true
	fail=1
else
	echo "Guard 1 PASS — wqm-storage-write is not reachable from mcp-server."
fi

echo "== Guard 3: no Qdrant-mutating symbol in read-only release binaries =="
# Build UNSTRIPPED: the release profile sets strip=true, which would erase the
# symbol table and make the scan meaningless. Override strip only for the scan.
# LTO stays on (default), so uncalled generic methods are dead-code-eliminated —
# the scan only sees symbols that are actually reachable.
cargo build --release --config 'profile.release.strip=false' -p mcp-server -p wqm-cli

# Pick a symbol reader: prefer llvm-nm, fall back to binutils nm.
if command -v llvm-nm >/dev/null 2>&1; then
	NM=llvm-nm
else
	NM=nm
fi

for bin in target/release/workspace-qdrant-mcp target/release/wqm; do
	if [ ! -f "$bin" ]; then
		echo "::error::Guard 3 FAILED — expected binary not built: $bin"
		fail=1
		continue
	fi
	# `nm` exits non-zero on a stripped/empty table; tolerate that, we grep output.
	hits="$("$NM" "$bin" 2>/dev/null | grep -E "$MUTATING" || true)"
	if [ -n "$hits" ]; then
		echo "::error::Guard 3 FAILED — Qdrant-mutating symbol reachable in $bin:"
		echo "$hits" | head -20
		fail=1
	else
		echo "Guard 3 PASS — no mutating symbol in $bin."
	fi
done

if [ "$fail" -ne 0 ]; then
	echo "storage-guards: one or more guards FAILED."
	exit 1
fi
echo "storage-guards: all guards PASS."
