#!/usr/bin/env bash
# test-restart-persistence.sh
#
# reference.yml bind-mounts memexd's state and Qdrant's storage under
# WQM_STATE_DIR. Nuking the containers with `compose down` (but not
# `--volumes`) should preserve both. After a fresh `up` the previously
# registered project must still be visible.

export TEST_NAME="restart-persistence"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

WQM_BIN="${WQM_BIN:-wqm}"
if ! command -v "${WQM_BIN}" >/dev/null 2>&1; then
	fail "wqm binary not found (looked for '${WQM_BIN}'); set WQM_BIN"
fi

project_dir="${WQM_DEV_ROOT}/restart-project"
mkdir -p "${project_dir}"
echo "# restart" >"${project_dir}/README.md"

export WQM_CLI_CONFIG="${TEST_SCRATCH}/cli-config.toml"
export WQM_PROFILE="docker-local"

# --- First boot: register + verify ----------------------------------------
log "first boot"
compose up -d --quiet-pull
poll_grpc "127.0.0.1:${MEMEXD_GRPC_PORT:-50051}"
poll_http qdrant "http://127.0.0.1:${QDRANT_HTTP_PORT:-6333}/readyz" 200

"${WQM_BIN}" project register "${project_dir}" >/dev/null 2>&1 || true
before=$("${WQM_BIN}" project list 2>&1)
assert_contains "${project_dir}" "${before}" "first-boot project list missing ${project_dir}"

# Give the daemon a moment to flush any pending WAL to disk.
sleep 3

# --- Stop without --volumes so bind mounts + named volumes survive --------
log "stopping stack (state preserved)"
compose down --timeout 10 >/dev/null

# Sanity: assert the bind-mounted state dir still exists on disk.
if [[ ! -d "${WQM_STATE_DIR}/memexd" ]]; then
	fail "memexd state dir ${WQM_STATE_DIR}/memexd disappeared after compose down"
fi
if [[ ! -d "${WQM_STATE_DIR}/qdrant/storage" ]]; then
	fail "qdrant storage dir ${WQM_STATE_DIR}/qdrant/storage disappeared"
fi

# --- Second boot: verify project still registered --------------------------
log "second boot"
compose up -d --quiet-pull
poll_grpc "127.0.0.1:${MEMEXD_GRPC_PORT:-50051}"
poll_http qdrant "http://127.0.0.1:${QDRANT_HTTP_PORT:-6333}/readyz" 200

after=$("${WQM_BIN}" project list 2>&1)
assert_contains "${project_dir}" "${after}" "second-boot project list lost ${project_dir}"

log "OK — projects + Qdrant storage persisted across compose down/up"
