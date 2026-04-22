#!/usr/bin/env bash
# test-stack-startup.sh
#
# Brings up the reference compose stack, waits for every service's health
# probe to report healthy, and verifies the container list matches the
# expected roster. Tears everything down on exit (success or failure).

export TEST_NAME="startup"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

log "launching reference stack"
compose up -d --quiet-pull

# Wait for each published endpoint to respond. These match the ports in
# docker/compose/reference.yml.
poll_http qdrant "http://127.0.0.1:${QDRANT_HTTP_PORT:-6333}/readyz" 200
poll_grpc "127.0.0.1:${MEMEXD_GRPC_PORT:-50051}"
poll_http memexd-metrics "http://127.0.0.1:${MEMEXD_METRICS_PORT:-9091}/health" 200
poll_http mcp-healthz "http://127.0.0.1:${MCP_HTTP_PORT:-6335}/healthz" 200

# Verify the three expected containers are in a running state.
services_running=$(compose ps --services --status=running | sort | tr '\n' ' ')
assert_contains "qdrant" "${services_running}" "qdrant service not running"
assert_contains "memexd" "${services_running}" "memexd service not running"
assert_contains "mcp" "${services_running}" "mcp service not running"

log "OK — all three services healthy"
