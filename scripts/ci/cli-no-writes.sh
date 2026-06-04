#!/usr/bin/env bash
# cli-no-writes.sh — CI guard (#82 WI-f5, task 35).
#
# The CLI must perform NO direct Qdrant or code-graph writes: the daemon is the
# single writer (ADR-003). The CLI may only request writes via daemon gRPC
# (enqueue / rebalance-idf / prune-logs RPCs) and may read SQLite or run SQLite
# UPDATEs through the WriteActor — those are NOT banned here.
#
# Banned symbols (direct write surface):
#   - StorageClient                    (the Qdrant write client)
#   - update_named_sparse_vectors / set_payload_on_point / delete_points /
#     delete_collection / upsert_points  (Qdrant writes)
#   - add_edge / add_node / upsert_nodes / upsert_edges  (code-graph writes)
#
# Excluded from the scan: the `bench`-gated benchmark module (dev-only) and test
# code. The ADR-002 `recover-state` offline path uses SQLite (not Qdrant/graph),
# so it is unaffected.
#
# Usage:  scripts/ci/cli-no-writes.sh [<repo_root>]
#         SELFTEST=1 scripts/ci/cli-no-writes.sh   # run the AC-f5.2 self-test
# Exit:   0 ok · 1 violation

set -uo pipefail

ROOT="${1:-.}"
CLI_SRC="$ROOT/src/rust/cli/src"

BANNED='StorageClient|update_named_sparse_vectors|set_payload_on_point|delete_points|delete_collection|upsert_points|\.add_edge\(|\.add_node\(|\.upsert_nodes\(|\.upsert_edges\('

# Scan a directory for banned write symbols, excluding bench + tests.
scan() {
	local dir="$1"
	grep -rnE "$BANNED" "$dir" --include='*.rs' 2>/dev/null |
		grep -v '/commands/benchmark/' |
		grep -vE '_tests?\.rs:' |
		grep -v '#\[cfg(test)\]' ||
		true
}

# AC-f5.2 negative test: a fixture with a banned symbol must be flagged, and a
# SQLite UPDATE must NOT be flagged.
if [[ "${SELFTEST:-0}" == "1" ]]; then
	echo "=== cli-no-writes self-test ==="
	tmp="$(mktemp -d)"
	trap 'rm -rf "$tmp"' EXIT
	mkdir -p "$tmp/src/rust/cli/src"
	printf 'use workspace_qdrant_core::storage::StorageClient;\n' >"$tmp/src/rust/cli/src/bad.rs"
	printf 'let _ = sqlx::query("UPDATE corpus_statistics SET last_corrected_n = 1");\n' >"$tmp/src/rust/cli/src/ok.rs"
	if [[ -z "$(scan "$tmp/src/rust/cli/src/bad.rs")" ]]; then
		echo "SELFTEST FAIL: banned StorageClient was not flagged"
		exit 1
	fi
	if [[ -n "$(scan "$tmp/src/rust/cli/src/ok.rs")" ]]; then
		echo "SELFTEST FAIL: a SQLite UPDATE was wrongly flagged"
		exit 1
	fi
	echo "SELFTEST OK: banned symbol flagged, SQLite UPDATE ignored"
	exit 0
fi

echo "=== CLI no-direct-writes check ==="
HITS="$(scan "$CLI_SRC")"
if [[ -n "$HITS" ]]; then
	echo "FAIL: CLI contains direct Qdrant/graph write surface:"
	echo "$HITS"
	echo "Route writes through the daemon (gRPC) — the daemon is the single writer."
	exit 1
fi
echo "OK: CLI has no direct Qdrant/graph write surface"
