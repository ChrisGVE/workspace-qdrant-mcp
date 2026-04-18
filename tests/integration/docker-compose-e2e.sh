#!/usr/bin/env bash
# tests/integration/docker-compose-e2e.sh
#
# End-to-end integration test for the workspace-qdrant-mcp telemetry pipeline.
#
# Boots: minimal.yml + observability.yml + qdrant.yml (CI overlay)
# Validates:
#   - All service health endpoints respond
#   - Prometheus scrapes all three jobs (memexd, mcp, qdrant)
#   - Core metrics are present and non-zero
#
# Requirements (available on ubuntu-latest GitHub Actions runners):
#   docker, docker compose (v2), curl, jq, timeout
#
# Usage:
#   tests/integration/docker-compose-e2e.sh
#
# Environment overrides:
#   MEMEXD_IMAGE   — memexd image ref (default: chrisgve/memexd:v0.1.3)
#   MCP_IMAGE      — MCP server image ref (default: chrisgve/workspace-qdrant-mcp:v0.1.3)
#   LOG_DIR        — where to dump container logs on failure (default: /tmp/wqm-e2e-logs)
#   POLL_TIMEOUT   — seconds to wait for each service (default: 120)
#   POLL_INTERVAL  — seconds between poll attempts (default: 2)

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
COMPOSE_DIR="${REPO_ROOT}/docker/compose"

export MEMEXD_IMAGE="${MEMEXD_IMAGE:-chrisgve/memexd:v0.1.3}"
export MCP_IMAGE="${MCP_IMAGE:-chrisgve/workspace-qdrant-mcp:v0.1.3}"
export MCP_SERVER_MODE="http"   # enable /metrics on :9092 for the test

LOG_DIR="${LOG_DIR:-/tmp/wqm-e2e-logs}"
POLL_TIMEOUT="${POLL_TIMEOUT:-120}"
POLL_INTERVAL="${POLL_INTERVAL:-2}"

COMPOSE_CMD="docker compose \
  -f ${COMPOSE_DIR}/minimal.yml \
  -f ${COMPOSE_DIR}/observability.yml \
  -f ${COMPOSE_DIR}/qdrant.yml"

# ── Helpers ────────────────────────────────────────────────────────────────────

log()  { echo "[$(date '+%H:%M:%S')] $*"; }
fail() { echo "[$(date '+%H:%M:%S')] FAIL: $*" >&2; exit 1; }

# poll_http <name> <url> <expected_http_code>
# Polls until the endpoint returns expected_http_code or POLL_TIMEOUT expires.
poll_http() {
  local name="$1"
  local url="$2"
  local expected="${3:-200}"
  local elapsed=0

  log "Waiting for ${name} at ${url} ..."
  while true; do
    local code
    code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 "${url}" 2>/dev/null || true)
    if [[ "${code}" == "${expected}" ]]; then
      log "${name} ready (HTTP ${code})"
      return 0
    fi
    if (( elapsed >= POLL_TIMEOUT )); then
      fail "${name} not ready after ${POLL_TIMEOUT}s (last HTTP code: ${code})"
    fi
    sleep "${POLL_INTERVAL}"
    elapsed=$(( elapsed + POLL_INTERVAL ))
  done
}

# prom_query <promql> — returns the Prometheus instant-query JSON result array
prom_query() {
  local query="$1"
  curl -sf -G "http://localhost:9090/api/v1/query" --data-urlencode "query=${query}" \
    2>/dev/null
}

# ── Cleanup ────────────────────────────────────────────────────────────────────

cleanup() {
  local exit_code=$?
  if (( exit_code != 0 )); then
    log "Non-zero exit (${exit_code}) — dumping logs to ${LOG_DIR}"
    mkdir -p "${LOG_DIR}"
    for svc in memexd workspace-qdrant-mcp qdrant wqm-prometheus wqm-grafana wqm-otel-collector; do
      docker logs "${svc}" >"${LOG_DIR}/${svc}.log" 2>&1 || true
    done
  fi
  log "Tearing down compose stack ..."
  ${COMPOSE_CMD} down --volumes --remove-orphans 2>/dev/null || true
}
trap cleanup EXIT

# ── Step 1: Boot stack ─────────────────────────────────────────────────────────

log "Starting stack: minimal + observability + qdrant ..."
log "  MEMEXD_IMAGE=${MEMEXD_IMAGE}"
log "  MCP_IMAGE=${MCP_IMAGE}"
${COMPOSE_CMD} up -d

# ── Step 2: Readiness polling ──────────────────────────────────────────────────

log "=== Polling service readiness ==="
poll_http "qdrant"      "http://localhost:6333/readyz"
poll_http "memexd"      "http://localhost:9091/health"
poll_http "mcp"         "http://localhost:9092/metrics"
poll_http "prometheus"  "http://localhost:9090/-/ready"
poll_http "grafana"     "http://localhost:3000/api/health"

# ── Step 3: MCP metric verification ───────────────────────────────────────────
#
# MCP invocation strategy: session-only metric assertion.
#
# The MCP server speaks stdio JSON-RPC (not HTTP RPC on port 50051).
# Driving a full tool-call over stdio from a shell script requires a
# dedicated MCP client binary not available in the default CI toolchain.
#
# Instead, we rely on the fact that `docker compose up` starts the MCP
# server container, which initialises the metrics process and registers
# wqm_mcp_session_count (value 0). Starting the container itself is
# sufficient to prove the metrics pipeline is wired up. The Prometheus
# assertion below confirms the metric is scrape-visible.
#
# Limitation: wqm_mcp_session_count will read 0 unless a real MCP client
# connects. The test asserts the metric *exists* (non-empty result set),
# not that its value is >0.

log "=== MCP: checking /metrics for wqm_mcp_ family ==="
mcp_metrics=$(curl -sf "http://localhost:9092/metrics" 2>/dev/null)
if ! echo "${mcp_metrics}" | grep -q "^wqm_mcp_"; then
  fail "No wqm_mcp_* metrics found in MCP /metrics output"
fi
log "wqm_mcp_* metrics present"

# ── Step 4: Prometheus target health ──────────────────────────────────────────

log "=== Prometheus: waiting for scrape targets to be UP ==="

# Give Prometheus time to complete its first scrape cycle (default 15s interval)
sleep 20

TARGETS_JSON=$(curl -sf "http://localhost:9090/api/v1/targets" 2>/dev/null)

check_target_up() {
  local job="$1"
  local health
  health=$(echo "${TARGETS_JSON}" | jq -r --arg j "${job}" \
    '.data.activeTargets[] | select(.labels.job==$j) | .health' 2>/dev/null | head -1)
  if [[ "${health}" != "up" ]]; then
    # Refresh — first snapshot might have been stale
    TARGETS_JSON=$(curl -sf "http://localhost:9090/api/v1/targets" 2>/dev/null)
    health=$(echo "${TARGETS_JSON}" | jq -r --arg j "${job}" \
      '.data.activeTargets[] | select(.labels.job==$j) | .health' 2>/dev/null | head -1)
  fi
  if [[ "${health}" != "up" ]]; then
    fail "Prometheus target '${job}' is not up (health=${health:-missing})"
  fi
  log "Target '${job}': up"
}

check_target_up "memexd"
check_target_up "mcp"
check_target_up "qdrant"

# ── Step 5: Metric value assertions ───────────────────────────────────────────

log "=== Prometheus metric assertions ==="

# up{job="memexd"} == 1
result=$(prom_query 'up{job="memexd"}')
value=$(echo "${result}" | jq -r '.data.result[0].value[1]' 2>/dev/null)
if [[ "${value}" != "1" ]]; then
  fail "up{job=\"memexd\"} expected 1, got: ${value}"
fi
log "up{job=\"memexd\"} == 1"

# memexd_uptime_seconds > 0
result=$(prom_query 'memexd_uptime_seconds')
value=$(echo "${result}" | jq -r '.data.result[0].value[1]' 2>/dev/null)
if [[ -z "${value}" ]] || (( $(echo "${value} <= 0" | bc -l) )); then
  fail "memexd_uptime_seconds expected > 0, got: ${value}"
fi
log "memexd_uptime_seconds == ${value} (> 0)"

# wqm_mcp_session_count: metric must exist (non-empty result set)
result=$(prom_query 'wqm_mcp_session_count')
count=$(echo "${result}" | jq '.data.result | length' 2>/dev/null)
if [[ -z "${count}" ]] || (( count == 0 )); then
  fail "wqm_mcp_session_count metric not found in Prometheus"
fi
log "wqm_mcp_session_count is present in Prometheus (${count} series)"

# ── All assertions passed ──────────────────────────────────────────────────────

log "=== All checks passed ==="
log "Stack: minimal + observability + qdrant"
log "Services: memexd, workspace-qdrant-mcp, qdrant, prometheus, grafana, otel-collector"
exit 0
