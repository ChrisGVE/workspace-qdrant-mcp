#!/usr/bin/env bash
# check-proto-consistency.sh
#
# Guards against gRPC proto drift between the canonical Rust daemon proto
# (the source of truth) and the TypeScript MCP daemon-client, which does NOT
# codegen from the proto — it hand-mirrors a subset and invokes RPCs via
# hardcoded camelCase method-name strings. If a daemon RPC is renamed or
# removed, the TS side still compiles and only fails at runtime with an
# "Unknown method" gRPC error. This check catches that at build time.
#
# Run from anywhere; paths are resolved relative to the repo root.
#
# Direction that matters (and is enforced): every RPC method-name string the
# TS client actually CALLS must correspond to an `rpc` defined in the canonical
# Rust proto. A TS call with no matching proto rpc => guaranteed runtime
# "Unknown method". (The reverse — proto rpcs the TS client does not call yet —
# is expected and benign, since the TS proto is an intentional subset.)
#
# Normalization: proto rpc names are PascalCase (e.g. NotifyServerStatus); the
# TS call strings are camelCase (e.g. notifyServerStatus). Some proto names also
# contain underscores in other generators, so we normalize BOTH sides by
# lowercasing and stripping underscores before comparing. After normalization
# `NotifyServerStatus` and `notifyServerStatus` both become `notifyserverstatus`.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXIT_CODE=0

PROTO_FILE="$REPO_ROOT/src/rust/daemon/proto/workspace_daemon.proto"
TS_DIR="$REPO_ROOT/src/typescript/mcp-server/src/clients/daemon-client"
TS_FILES=("$TS_DIR/system-methods.ts" "$TS_DIR/service-methods.ts")

echo "=== Proto Consistency Check ==="
echo ""

# ── Locate inputs ──────────────────────────────────────────────────────

if [ ! -f "$PROTO_FILE" ]; then
    echo "  ERROR: canonical proto not found at $PROTO_FILE"
    echo "=== Proto consistency checks FAILED ==="
    exit 1
fi

for f in "${TS_FILES[@]}"; do
    if [ ! -f "$f" ]; then
        echo "  ERROR: TS daemon-client file not found at $f"
        echo "=== Proto consistency checks FAILED ==="
        exit 1
    fi
done

# Normalize a name: lowercase, strip underscores.
normalize() {
    tr '[:upper:]' '[:lower:]' | tr -d '_'
}

# ── Extract canonical RPC names from the Rust proto ────────────────────
# Match `rpc <Name>(` and capture <Name>.

# `|| true` so a parse-miss falls through to the friendly empty-guard below
# instead of aborting bare under `set -euo pipefail`.
PROTO_RPCS_RAW=$(grep -oE '^[[:space:]]*rpc[[:space:]]+[A-Za-z_][A-Za-z0-9_]*' "$PROTO_FILE" \
    | sed -E 's/^[[:space:]]*rpc[[:space:]]+//' || true)

if [ -z "$PROTO_RPCS_RAW" ]; then
    echo "  ERROR: no 'rpc' definitions parsed from $PROTO_FILE"
    echo "=== Proto consistency checks FAILED ==="
    exit 1
fi

PROTO_COUNT=$(echo "$PROTO_RPCS_RAW" | wc -l | tr -d ' ')
PROTO_NORM=$(echo "$PROTO_RPCS_RAW" | normalize | sort -u)

echo "--- Canonical proto ---"
echo "  $PROTO_FILE"
echo "  parsed $PROTO_COUNT rpc method(s)"
echo ""

# ── Extract RPC method-name strings the TS client calls ────────────────
# Each call site is grpcUnaryWithTimeout(<client>, '<methodName>', ...). The
# <client> is always `this.<something>Client`. We capture the FIRST quoted
# string literal that follows a `this.<word>Client` token — that is the
# methodName (2nd positional arg), never the optional 5th operationName label.
#
# Handles both single-line calls (this.systemClient, 'getStatus', {}, ...) and
# multi-line calls where the client and method-name string are on separate
# lines. We collapse the two TS files to a single whitespace-normalized stream
# so the regex spans line breaks.

# `|| true` so a parse-miss falls through to the friendly empty-guard below
# instead of aborting bare under `set -euo pipefail`.
TS_CALLS_RAW=$(cat "${TS_FILES[@]}" \
    | tr '\n' ' ' \
    | grep -oE "this\.[A-Za-z]+Client,[[:space:]]*'[A-Za-z0-9_]+'" \
    | sed -E "s/.*,[[:space:]]*'([A-Za-z0-9_]+)'/\1/" || true)

if [ -z "$TS_CALLS_RAW" ]; then
    echo "  ERROR: no grpcUnaryWithTimeout RPC call strings parsed from TS client"
    echo "=== Proto consistency checks FAILED ==="
    exit 1
fi

TS_CALLS_UNIQUE=$(echo "$TS_CALLS_RAW" | sort -u)
TS_COUNT=$(echo "$TS_CALLS_UNIQUE" | wc -l | tr -d ' ')

echo "--- TS daemon-client calls ---"
echo "  ${TS_FILES[0]}"
echo "  ${TS_FILES[1]}"
echo "  found $TS_COUNT distinct RPC call string(s)"
echo ""

# ── Compare: every TS call must exist in the proto ─────────────────────

echo "--- TS calls vs canonical proto ---"

ORPHANS=()
while IFS= read -r call; do
    [ -z "$call" ] && continue
    norm=$(echo "$call" | normalize)
    if ! echo "$PROTO_NORM" | grep -qx "$norm"; then
        ORPHANS+=("$call")
    fi
done <<< "$TS_CALLS_UNIQUE"

if [ ${#ORPHANS[@]} -gt 0 ]; then
    echo ""
    echo "  ERROR: TS daemon-client calls RPC(s) with no matching 'rpc' in the canonical proto!"
    echo "  These will fail at runtime with a gRPC 'Unknown method' error:"
    for o in "${ORPHANS[@]}"; do
        echo "    - $o"
    done
    echo ""
    echo "  Either the proto rpc was renamed/removed, or the TS call string is wrong."
    echo "  Fix the TS call in $TS_DIR/ or restore the rpc in:"
    echo "    $PROTO_FILE"
    EXIT_CODE=1
else
    echo "  OK: all $TS_COUNT TS RPC call(s) map to a canonical proto rpc"
fi
echo ""

# ── Summary ────────────────────────────────────────────────────────────

if [ $EXIT_CODE -eq 0 ]; then
    echo "=== All proto consistency checks passed ==="
else
    echo "=== Proto consistency checks FAILED ==="
fi

exit $EXIT_CODE
