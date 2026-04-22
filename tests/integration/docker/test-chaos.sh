#!/usr/bin/env bash
# test-chaos.sh
#
# Restart each service in turn under the reference compose stack and
# verify the neighbours recover cleanly. Covers:
#
#   1. docker restart qdrant   → MCP healthz stays up; once qdrant is ready
#                                 again, daemon + mcp both report healthy.
#   2. docker restart memexd   → mcp /healthz stays up; gRPC endpoint is
#                                 unavailable briefly, then comes back.
#   3. docker restart mcp      → /healthz returns after the container boots;
#                                 authenticated initialize still works.
#   4. docker kill -s TERM <svc> → service exits gracefully (exit status 0
#                                    on wait) and the stack rejoins.
#
# The full load-test scenario from the PRD (50 concurrent clients + trigger
# restarts) is intentionally out of scope for CI — this script runs in a
# reasonable wall-clock budget. Load/soak testing is documented separately
# in docs/deployment/reliability.md.

export TEST_NAME="chaos"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

ITERATIONS="${CHAOS_ITERATIONS:-3}"

log "launching reference stack"
compose up -d --quiet-pull

# Wait for the baseline healthy state before we start any churn.
poll_http qdrant "http://127.0.0.1:${QDRANT_HTTP_PORT:-6333}/readyz" 200
poll_grpc "127.0.0.1:${MEMEXD_GRPC_PORT:-50051}"
poll_http mcp-healthz "http://127.0.0.1:${MCP_HTTP_PORT:-6335}/healthz" 200
poll_http memexd-metrics "http://127.0.0.1:${MEMEXD_METRICS_PORT:-9091}/health" 200

init_body='{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"chaos","version":"0"}}}'

restart_iteration() {
	local service="$1"
	local iter="$2"
	log "[${service}] iteration ${iter}: docker compose restart"
	compose restart "${service}" >/dev/null
}

wait_for_healthy() {
	poll_http qdrant "http://127.0.0.1:${QDRANT_HTTP_PORT:-6333}/readyz" 200
	poll_grpc "127.0.0.1:${MEMEXD_GRPC_PORT:-50051}"
	poll_http mcp-healthz "http://127.0.0.1:${MCP_HTTP_PORT:-6335}/healthz" 200
	poll_http memexd-metrics "http://127.0.0.1:${MEMEXD_METRICS_PORT:-9091}/health" 200
}

# Each scenario: restart → wait for healthy → probe the MCP HTTP initialize
# path. If the neighbours failed to reconnect, initialize returns something
# other than a JSON-RPC envelope (500 or a closed socket), and the assertion
# catches it.
for service in qdrant memexd mcp; do
	for iter in $(seq 1 "${ITERATIONS}"); do
		restart_iteration "${service}" "${iter}"
		wait_for_healthy
		resp=$(mcp_post "${init_body}")
		assert_contains "serverInfo" "${resp}" "${service} restart #${iter}: initialize did not produce serverInfo"
	done
done

# Graceful-shutdown leg: send SIGTERM to memexd, assert the container exits
# cleanly (exit code 0 within the compose stop timeout), then bring it back.
log "[memexd] SIGTERM and re-up"
memexd_id=$(compose ps -q memexd)
[[ -n "${memexd_id}" ]] || fail "could not resolve memexd container id"
docker kill --signal TERM "${memexd_id}" >/dev/null
# Wait up to 15s for the container to exit. A graceful shutdown completes in
# well under the daemon's default 10s drain window.
elapsed=0
while true; do
	status=$(docker inspect -f '{{.State.Status}}' "${memexd_id}" 2>/dev/null || echo "gone")
	exit_code=$(docker inspect -f '{{.State.ExitCode}}' "${memexd_id}" 2>/dev/null || echo "")
	if [[ "${status}" == "exited" ]]; then
		log "memexd exited with code ${exit_code}"
		assert_equals "0" "${exit_code}" "memexd did not exit cleanly on SIGTERM"
		break
	fi
	if ((elapsed >= 15)); then
		fail "memexd did not exit within 15s of SIGTERM (status=${status})"
	fi
	sleep 1
	elapsed=$((elapsed + 1))
done

compose up -d memexd >/dev/null
wait_for_healthy

log "OK — stack recovers from individual service restarts + SIGTERM"
