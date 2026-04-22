#!/usr/bin/env bash
# test-mcp-http.sh
#
# Exercises the MCP Streamable HTTP transport end-to-end:
#   1. /healthz is open and returns 200.
#   2. /mcp without a bearer token returns 401.
#   3. /mcp with the wrong token returns 401.
#   4. /mcp with the correct token initializes the session and lists tools.
#
# Does NOT invoke the full tool pipeline (store/search/retrieve). Those need
# live data and are deferred to test-path-transparency.sh.

export TEST_NAME="mcp-http"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

log "launching reference stack"
compose up -d --quiet-pull

poll_http mcp-healthz "http://127.0.0.1:${MCP_HTTP_PORT:-6335}/healthz" 200

# --- Unauthenticated request → 401 -----------------------------------------
log "expecting 401 with no bearer token"
code=$(curl -sk -o /dev/null -w '%{http_code}' --max-time 10 \
	-X POST \
	-H "Content-Type: application/json" \
	-d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"e2e","version":"0"}}}' \
	"http://127.0.0.1:${MCP_HTTP_PORT:-6335}/mcp")
assert_equals "401" "${code}" "unauthenticated /mcp should return 401"

# --- Wrong token → 401 ------------------------------------------------------
log "expecting 401 with a bogus token"
code=$(curl -sk -o /dev/null -w '%{http_code}' --max-time 10 \
	-X POST \
	-H "Authorization: Bearer definitely-not-the-token" \
	-H "Content-Type: application/json" \
	-d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"e2e","version":"0"}}}' \
	"http://127.0.0.1:${MCP_HTTP_PORT:-6335}/mcp")
assert_equals "401" "${code}" "bogus-token /mcp should return 401"

# --- Valid initialize ------------------------------------------------------
log "initializing MCP session with correct bearer token"
init_body='{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"e2e","version":"0"}}}'
init_resp=$(mcp_post "${init_body}")
assert_contains "serverInfo" "${init_resp}" "initialize response missing serverInfo"
assert_contains "workspace-qdrant-mcp" "${init_resp}" "initialize response missing server name"

# --- tools/list ------------------------------------------------------------
log "requesting tools/list"
list_body='{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}'
list_resp=$(mcp_post "${list_body}")
# Streamable HTTP may return SSE — just check every MCP tool name shows up in
# the raw payload.
for tool in store search rules retrieve grep list; do
	assert_contains "\"${tool}\"" "${list_resp}" "tools/list missing ${tool}"
done

log "OK — MCP HTTP transport handles auth + initialize + tools/list"
