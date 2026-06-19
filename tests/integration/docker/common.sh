#!/usr/bin/env bash
# tests/integration/docker/common.sh
#
# Shared helpers for the reference-compose integration suite. Source this
# from every test script — it sets `set -euo pipefail`, exports paths,
# installs a cleanup trap that tears the compose project down, and provides
# polling + assertion helpers.
#
# Each caller must export:
#   TEST_NAME          — unique short name, used in the compose project prefix
#                        and log-dump directory
#
# The caller may override:
#   POLL_TIMEOUT       — seconds to wait for each service (default: 180)
#   POLL_INTERVAL      — seconds between poll attempts (default: 3)
#   MEMEXD_IMAGE, MCP_IMAGE, QDRANT_VERSION — see compose/reference.yml
#   WQM_DEV_ROOT       — host path to bind-mount as watch root
#                        (default: scratch dir inside BATS_TMPDIR)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
COMPOSE_DIR="${REPO_ROOT}/docker/compose"

TEST_NAME="${TEST_NAME:-wqm-ref}"
COMPOSE_PROJECT="wqm-e2e-${TEST_NAME}"
POLL_TIMEOUT="${POLL_TIMEOUT:-180}"
POLL_INTERVAL="${POLL_INTERVAL:-3}"

# Scratch directory for everything this test writes (compose state, dev root).
TEST_SCRATCH="${TEST_SCRATCH:-$(mktemp -d "/tmp/wqm-e2e-${TEST_NAME}.XXXXXX")}"
export TEST_SCRATCH

export WQM_DEV_ROOT="${WQM_DEV_ROOT:-${TEST_SCRATCH}/dev-root}"
mkdir -p "${WQM_DEV_ROOT}"
export WQM_STATE_DIR="${WQM_STATE_DIR:-${TEST_SCRATCH}/state}"
# reference.yml bind-mounts ${WQM_STATE_DIR}/memexd onto the daemon's XDG data
# home. Create that subdir here: if Docker auto-creates a missing bind source it
# does so as root, and the container's unprivileged memexd user (uid 1000) then
# can't open state.db → SQLITE_CANTOPEN. The host uid in CI (runner) also differs
# from 1000, so make the ephemeral scratch tree world-writable to bridge the
# bind-mount uid gap. Scratch is per-run and discarded, so 0777 is safe here.
mkdir -p "${WQM_STATE_DIR}/memexd"
chmod -R 0777 "${WQM_STATE_DIR}"

# Bearer token generated per-run so two runs in parallel don't collide.
export MCP_HTTP_TOKEN="${MCP_HTTP_TOKEN:-$(head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n')}"

# Image tags — override in CI if the workflow wants a specific tag.
export MEMEXD_IMAGE="${MEMEXD_IMAGE:-ghcr.io/chrisgve/memexd:latest}"
export MCP_IMAGE="${MCP_IMAGE:-ghcr.io/chrisgve/workspace-qdrant-mcp:latest}"
export QDRANT_VERSION="${QDRANT_VERSION:-latest}"
export WQM_VERSION="${WQM_VERSION:-latest}"

LOG_DIR="${LOG_DIR:-${TEST_SCRATCH}/logs}"
mkdir -p "${LOG_DIR}"

log() { echo "[$(date '+%H:%M:%S')] ${TEST_NAME}: $*"; }
fail() {
	echo "[$(date '+%H:%M:%S')] ${TEST_NAME}: FAIL: $*" >&2
	exit 1
}

compose() {
	docker compose \
		--project-name "${COMPOSE_PROJECT}" \
		-f "${COMPOSE_DIR}/reference.yml" \
		"$@"
}

cleanup() {
	local exit_code=$?
	log "cleanup: docker compose down (exit=${exit_code})"
	# Dump container logs if the run failed so CI has something to look at.
	if [[ ${exit_code} -ne 0 ]]; then
		log "dumping logs to ${LOG_DIR}"
		compose logs --no-color >"${LOG_DIR}/compose.log" 2>&1 || true
		compose ps >"${LOG_DIR}/compose-ps.log" 2>&1 || true
	fi
	compose down --volumes --remove-orphans --timeout 5 >/dev/null 2>&1 || true
	# Leave TEST_SCRATCH behind on failure; clean it up on success.
	if [[ ${exit_code} -eq 0 ]]; then
		rm -rf "${TEST_SCRATCH}"
	else
		log "scratch dir preserved at ${TEST_SCRATCH}"
	fi
	exit ${exit_code}
}

trap cleanup EXIT INT TERM

# poll_http <name> <url> [<expected_status>]
# Polls until curl returns the expected HTTP code or POLL_TIMEOUT elapses.
poll_http() {
	local name="$1"
	local url="$2"
	local expected="${3:-200}"
	local elapsed=0
	log "waiting for ${name} at ${url} (expect ${expected})"
	while true; do
		local code
		code="$(curl -sk -o /dev/null -w '%{http_code}' --max-time 5 "${url}" 2>/dev/null || true)"
		if [[ "${code}" == "${expected}" ]]; then
			log "${name} responded ${code}"
			return 0
		fi
		if ((elapsed >= POLL_TIMEOUT)); then
			fail "${name} did not return ${expected} within ${POLL_TIMEOUT}s (last code: ${code:-none})"
		fi
		sleep "${POLL_INTERVAL}"
		elapsed=$((elapsed + POLL_INTERVAL))
	done
}

# poll_grpc <host:port>
# Succeeds as soon as the TCP port accepts a connection.
poll_grpc() {
	local addr="$1"
	local elapsed=0
	log "waiting for gRPC listener at ${addr}"
	while true; do
		if (exec 3<>"/dev/tcp/${addr%%:*}/${addr##*:}") 2>/dev/null; then
			exec 3<&-
			log "${addr} accepted connection"
			return 0
		fi
		if ((elapsed >= POLL_TIMEOUT)); then
			fail "${addr} never accepted a connection (${POLL_TIMEOUT}s)"
		fi
		sleep "${POLL_INTERVAL}"
		elapsed=$((elapsed + POLL_INTERVAL))
	done
}

# assert_equals <expected> <actual> <message>
assert_equals() {
	local expected="$1"
	local actual="$2"
	local msg="${3:-mismatch}"
	if [[ "${expected}" != "${actual}" ]]; then
		fail "${msg}: expected ${expected}, got ${actual}"
	fi
}

# assert_contains <needle> <haystack> <message>
assert_contains() {
	local needle="$1"
	local haystack="$2"
	local msg="${3:-substring missing}"
	if [[ "${haystack}" != *"${needle}"* ]]; then
		fail "${msg}: ${needle} not in ${haystack}"
	fi
}

# mcp_post <json_body>
# Performs an authenticated POST to /mcp and prints the raw response body.
mcp_post() {
	local body="$1"
	curl -sk \
		--max-time 30 \
		-H "Authorization: Bearer ${MCP_HTTP_TOKEN}" \
		-H "Content-Type: application/json" \
		-H "Accept: application/json, text/event-stream" \
		-d "${body}" \
		"http://127.0.0.1:${MCP_HTTP_PORT:-6335}/mcp"
}
