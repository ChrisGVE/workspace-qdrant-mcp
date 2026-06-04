#!/usr/bin/env bash
# cli-no-core.sh — CI guard (#82 WI-e3, task 33).
#
# The DEFAULT `wqm-cli` build must NOT link `workspace-qdrant-core`: clients talk
# to the daemon over gRPC (wqm-common + wqm-client + wqm-proto). Core is a
# dev-only edge, gated behind the `bench` feature.
#
# Checks:
#   1. `cargo tree -p wqm-cli` (DEFAULT) contains NO workspace-qdrant-core.
#   2. `cargo tree -p wqm-cli --features bench` DOES (positive control).
#
# Usage: scripts/ci/cli-no-core.sh [<repo_root>]
# Exit: 0 ok · 1 violation · 2 tooling error

set -uo pipefail

ROOT="${1:-.}"
if [[ -d "$ROOT/src/rust" ]]; then
	cd "$ROOT/src/rust"
else
	cd "$ROOT"
fi

echo "=== CLI core-isolation check ==="

DEFAULT_TREE="$(cargo tree -p wqm-cli 2>/dev/null)" || {
	echo "ERROR: 'cargo tree -p wqm-cli' failed"
	exit 2
}

if grep -q "workspace-qdrant-core" <<<"$DEFAULT_TREE"; then
	echo "FAIL: the default wqm-cli build links workspace-qdrant-core:"
	grep "workspace-qdrant-core" <<<"$DEFAULT_TREE"
	exit 1
fi
echo "OK: default wqm-cli does not depend on workspace-qdrant-core"

BENCH_TREE="$(cargo tree -p wqm-cli --features bench 2>/dev/null)" || {
	echo "ERROR: 'cargo tree -p wqm-cli --features bench' failed"
	exit 2
}
if ! grep -q "workspace-qdrant-core" <<<"$BENCH_TREE"; then
	echo "FAIL: '--features bench' should link workspace-qdrant-core but does not"
	exit 1
fi
echo "OK: --features bench links workspace-qdrant-core (sanctioned dev-only edge)"

echo "=== CLI core-isolation check passed ==="
