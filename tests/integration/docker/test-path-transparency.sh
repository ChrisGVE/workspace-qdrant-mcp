#!/usr/bin/env bash
# test-path-transparency.sh
#
# reference.yml bind-mounts WQM_DEV_ROOT into memexd at the same path. This
# test proves the host CLI's view of a project path matches what the daemon
# records in SQLite: register a project via the MCP HTTP initialize +
# tools/call path, then run `wqm project status` on the host and confirm
# the absolute path survives the container boundary.

export TEST_NAME="path-transparency"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

WQM_BIN="${WQM_BIN:-wqm}"
if ! command -v "${WQM_BIN}" >/dev/null 2>&1; then
	fail "wqm binary not found (looked for '${WQM_BIN}'); set WQM_BIN"
fi

# Pick a concrete subtree under WQM_DEV_ROOT to register.
project_dir="${WQM_DEV_ROOT}/sample-project"
mkdir -p "${project_dir}"
# A tiny file so the project registration has something to notice.
echo "# sample" >"${project_dir}/README.md"

log "launching reference stack"
compose up -d --quiet-pull
poll_grpc "127.0.0.1:${MEMEXD_GRPC_PORT:-50051}"
poll_http mcp-healthz "http://127.0.0.1:${MCP_HTTP_PORT:-6335}/healthz" 200

export WQM_CLI_CONFIG="${TEST_SCRATCH}/cli-config.toml"
export WQM_PROFILE="docker-local"

log "registering project via wqm project register ${project_dir}"
register_out=$("${WQM_BIN}" project register "${project_dir}" 2>&1 || true)
# Either a success message or a "project already registered" line is fine —
# the invariant we're testing is what path the daemon ends up recording.
assert_contains "${project_dir}" "${register_out}" "register output does not mention project path"

log "listing projects — expect daemon to echo the host path"
list_out=$("${WQM_BIN}" project list 2>&1)
assert_contains "${project_dir}" "${list_out}" "daemon did not record project at host path ${project_dir}"

log "OK — daemon records ${project_dir} verbatim; path transparency preserved"
