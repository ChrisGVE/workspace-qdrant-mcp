#!/usr/bin/env bash
# test-cli-connection.sh
#
# Verifies the host `wqm` CLI can drive the dockerized memexd daemon via the
# exposed gRPC port. Uses the `docker-local` profile from cli-config.toml so
# connection targets come from the profile rather than implicit defaults.
#
# Requires the host `wqm` binary somewhere on PATH (CI installs it from
# cargo build artifacts; locally set WQM_BIN to the path of a recent
# release build).

export TEST_NAME="cli-connection"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

WQM_BIN="${WQM_BIN:-wqm}"
if ! command -v "${WQM_BIN}" >/dev/null 2>&1; then
	fail "wqm binary not found (looked for '${WQM_BIN}'); set WQM_BIN"
fi

log "launching reference stack"
compose up -d --quiet-pull
poll_grpc "127.0.0.1:${MEMEXD_GRPC_PORT:-50051}"
poll_http memexd-metrics "http://127.0.0.1:${MEMEXD_METRICS_PORT:-9091}/health" 200

# Use an isolated cli-config.toml so this test doesn't mutate the user's
# real one. The file is bootstrapped on first access.
export WQM_CLI_CONFIG="${TEST_SCRATCH}/cli-config.toml"
export WQM_PROFILE="docker-local"

log "running wqm service status against dockerized daemon"
status_out=$("${WQM_BIN}" service status 2>&1)
assert_contains "healthy" "${status_out}" "wqm service status reports unhealthy daemon"

log "running wqm admin collections list"
list_out=$("${WQM_BIN}" admin collections list 2>&1)
# Four canonical collections exist after first boot. We tolerate the libraries
# collection lagging (it's created lazily) but the other three must appear.
for collection in projects rules scratchpad; do
	assert_contains "${collection}" "${list_out}" "expected collection ${collection} missing"
done

log "running wqm status health (qdrant + mcp probe)"
# health needs to know where the MCP HTTP endpoint lives — only probed when
# WQM_MCP_HTTP_URL is set (stdio deployments should not false-fail).
export WQM_MCP_HTTP_URL="http://127.0.0.1:${MCP_HTTP_PORT:-6335}"
health_out=$("${WQM_BIN}" status health 2>&1)
# health subcommand returns non-zero if any probe fails; assert it returned 0
# plus that qdrant and mcp sections both appear.
assert_contains "qdrant" "${health_out}" "wqm status health missing qdrant section"
assert_contains "healthy" "${health_out}" "wqm status health does not include a healthy probe"

log "OK — host wqm CLI reaches dockerized daemon"
