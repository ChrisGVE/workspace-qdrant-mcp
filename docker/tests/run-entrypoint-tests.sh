#!/bin/bash
# docker/tests/run-entrypoint-tests.sh
#
# Integration tests for docker/memexd-entrypoint.sh (spec 16 §9.1).
#
# These tests do NOT require Docker — they exercise the entrypoint
# script with synthesised config / override / mountinfo fixtures and
# stubbed paths via WQM_OVERRIDE_PATH / WQM_CONFIG_PATH /
# WQM_MOUNTINFO_PATH / WQM_MEMEXD_BIN / WQM_ENTRYPOINT_SKIP_EXEC.
#
# The full Docker image build is Phase C scope (multi-arch with ONNX
# Linux static lib + memexd binary build). The image entrypoint logic
# tested here is fully independent of the image build.
#
# Usage:
#   docker/tests/run-entrypoint-tests.sh
#
# Exit:
#   0 — all scenarios passed
#   1 — at least one scenario failed
#
# Requires: bash 4+, python3 with PyYAML installed (matches the runtime
# image dependencies).

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ENTRYPOINT="${REPO_ROOT}/docker/memexd-entrypoint.sh"

if [ ! -x "${ENTRYPOINT}" ]; then
	echo "FAIL: entrypoint not executable: ${ENTRYPOINT}" >&2
	exit 1
fi

# Exit codes from the entrypoint.
readonly EXIT_HASH_MISMATCH=10
readonly EXIT_MOUNT_MISSING=11

# Counter state.
PASS=0
FAIL=0
FAILURES=()

# Run one scenario.
#
# Args:
#   $1 — human-readable test name
#   $2 — expected exit code
#   $3 — required substring in entrypoint stderr (or empty to skip check)
#   $4 — forbidden substring in entrypoint stderr (or empty to skip check)
#   remaining — env-prefixed command, e.g. WQM_OVERRIDE_PATH=... WQM_CONFIG_PATH=...
run_test() {
	local name="$1"
	local want_exit="$2"
	local must_contain="$3"
	local must_not_contain="$4"
	shift 4

	local stderr_file
	stderr_file=$(mktemp)
	local got_exit=0
	env "$@" \
		WQM_ENTRYPOINT_SKIP_EXEC=1 \
		WQM_MEMEXD_BIN=/bin/true \
		bash "${ENTRYPOINT}" --foreground 2>"${stderr_file}" >/dev/null
	got_exit=$?

	local ok=1
	if [ "${got_exit}" -ne "${want_exit}" ]; then
		ok=0
	fi
	if [ -n "${must_contain}" ] && ! grep -q -F "${must_contain}" "${stderr_file}"; then
		ok=0
	fi
	if [ -n "${must_not_contain}" ] && grep -q -F "${must_not_contain}" "${stderr_file}"; then
		ok=0
	fi

	if [ "${ok}" -eq 1 ]; then
		printf '  PASS  %s\n' "${name}"
		PASS=$((PASS + 1))
	else
		printf '  FAIL  %s (exit=%s want=%s)\n' "${name}" "${got_exit}" "${want_exit}"
		sed 's/^/        /' "${stderr_file}" >&2
		FAILURES+=("${name}")
		FAIL=$((FAIL + 1))
	fi
	rm -f "${stderr_file}"
}

# Compute the spec hash from a list of (host, container) pairs.
# Args: pairs given as repeated "host:container" strings.
spec_hash() {
	python3 - "$@" <<'PY'
import hashlib, sys, yaml
entries = []
for arg in sys.argv[1:]:
    host, _, container = arg.partition(":")
    entries.append({"host": host, "container": container})
ser = yaml.safe_dump(entries, default_flow_style=False, sort_keys=False)
sys.stdout.write(hashlib.sha256(ser.encode("utf-8")).hexdigest())
PY
}

# Scratch dir for fixtures.
WORK=$(mktemp -d)
trap 'rm -rf "${WORK}"' EXIT

echo "==> stale-override scenario (subtask 11.15)"
# Build a config + a MATCHING override, then mutate config to make hash drift.
mkdir -p "${WORK}/stale/c1"
cat >"${WORK}/stale/config.yaml" <<EOF
mounts:
  - host: /irrelevant
    container: ${WORK}/stale/c1
EOF
HASH_BEFORE=$(spec_hash "/irrelevant:${WORK}/stale/c1")
cat >"${WORK}/stale/override.yaml" <<EOF
# wqm-config-hash: ${HASH_BEFORE}
services:
  memexd:
    volumes:
      - "/irrelevant:${WORK}/stale/c1"
EOF
# Round 1: matching → must pass layer 1.
run_test "matching hash passes layer 1" 0 \
	"layer 1: ok" \
	"hash mismatch" \
	"WQM_OVERRIDE_PATH=${WORK}/stale/override.yaml" \
	"WQM_CONFIG_PATH=${WORK}/stale/config.yaml" \
	"WQM_MOUNTINFO_PATH=${WORK}/nonexistent"
# Round 2: mutate config, hash drifts.
cat >"${WORK}/stale/config.yaml" <<EOF
mounts:
  - host: /changed
    container: ${WORK}/stale/c1
EOF
run_test "stale override aborts with hash mismatch" "${EXIT_HASH_MISMATCH}" \
	"stale (hash mismatch)" \
	"" \
	"WQM_OVERRIDE_PATH=${WORK}/stale/override.yaml" \
	"WQM_CONFIG_PATH=${WORK}/stale/config.yaml" \
	"WQM_MOUNTINFO_PATH=${WORK}/nonexistent"
# Round 3: regenerate the override and ensure pass again.
HASH_AFTER=$(spec_hash "/changed:${WORK}/stale/c1")
cat >"${WORK}/stale/override.yaml" <<EOF
# wqm-config-hash: ${HASH_AFTER}
services:
  memexd: {}
EOF
run_test "regenerated override passes layer 1" 0 \
	"layer 1: ok" \
	"hash mismatch" \
	"WQM_OVERRIDE_PATH=${WORK}/stale/override.yaml" \
	"WQM_CONFIG_PATH=${WORK}/stale/config.yaml" \
	"WQM_MOUNTINFO_PATH=${WORK}/nonexistent"

echo "==> missing-mount scenario (subtask 11.16)"
# Config declares two mounts; only one is present on disk.
mkdir -p "${WORK}/missing/present"
cat >"${WORK}/missing/config.yaml" <<EOF
mounts:
  - host: /irrelevant1
    container: ${WORK}/missing/present
  - host: /irrelevant2
    container: ${WORK}/missing/absent
EOF
HASH_M=$(spec_hash "/irrelevant1:${WORK}/missing/present" "/irrelevant2:${WORK}/missing/absent")
cat >"${WORK}/missing/override.yaml" <<EOF
# wqm-config-hash: ${HASH_M}
services:
  memexd: {}
EOF
run_test "missing container path aborts layer 2" "${EXIT_MOUNT_MISSING}" \
	"Required mount missing: ${WORK}/missing/absent" \
	"" \
	"WQM_OVERRIDE_PATH=${WORK}/missing/override.yaml" \
	"WQM_CONFIG_PATH=${WORK}/missing/config.yaml" \
	"WQM_MOUNTINFO_PATH=${WORK}/nonexistent"
# Now create the missing dir; check layer 2 passes.
mkdir -p "${WORK}/missing/absent"
run_test "creating the missing dir lets layer 2 pass" 0 \
	"layer 2: ok (2 mount(s) verified present)" \
	"" \
	"WQM_OVERRIDE_PATH=${WORK}/missing/override.yaml" \
	"WQM_CONFIG_PATH=${WORK}/missing/config.yaml" \
	"WQM_MOUNTINFO_PATH=${WORK}/nonexistent"

echo "==> spurious-mount scenario (subtask 11.17)"
# Set up two expected mounts + a synthesised mountinfo containing an extra
# bind mount the entrypoint must warn about (non-fatal).
mkdir -p "${WORK}/spurious/c1" "${WORK}/spurious/c2"
cat >"${WORK}/spurious/config.yaml" <<EOF
mounts:
  - host: /irrelevant1
    container: ${WORK}/spurious/c1
  - host: /irrelevant2
    container: ${WORK}/spurious/c2
EOF
HASH_S=$(spec_hash "/irrelevant1:${WORK}/spurious/c1" "/irrelevant2:${WORK}/spurious/c2")
cat >"${WORK}/spurious/override.yaml" <<EOF
# wqm-config-hash: ${HASH_S}
services:
  memexd: {}
EOF
cat >"${WORK}/spurious/mountinfo" <<EOF
22 21 0:21 / /proc rw - proc proc rw
23 21 0:22 / /sys rw - sysfs sysfs rw
36 21 0:36 / ${WORK}/spurious/c1 rw - tmpfs tmpfs rw
37 21 0:37 / ${WORK}/spurious/c2 rw - tmpfs tmpfs rw
40 21 0:40 / /etc/wqm/config.yaml rw - tmpfs tmpfs rw
41 21 0:41 / /var/lib/wqm rw - tmpfs tmpfs rw
50 21 0:50 / /mnt/scratch-debug rw - tmpfs tmpfs rw
EOF
run_test "spurious mount emits warning but does not abort" 0 \
	"Unexpected bind mount detected: /mnt/scratch-debug" \
	"" \
	"WQM_OVERRIDE_PATH=${WORK}/spurious/override.yaml" \
	"WQM_CONFIG_PATH=${WORK}/spurious/config.yaml" \
	"WQM_MOUNTINFO_PATH=${WORK}/spurious/mountinfo"

# Octal-escape decoding case (spec §9.1 — mountinfo with literal spaces).
cat >"${WORK}/spurious/mountinfo-escaped" <<EOF
36 21 0:36 / ${WORK}/spurious/c1 rw - tmpfs tmpfs rw
37 21 0:37 / ${WORK}/spurious/c2 rw - tmpfs tmpfs rw
55 21 0:55 / /mnt/path\\040with\\040spaces rw - tmpfs tmpfs rw
EOF
run_test "spurious mount with octal-escaped space decodes" 0 \
	"Unexpected bind mount detected: /mnt/path with spaces" \
	"" \
	"WQM_OVERRIDE_PATH=${WORK}/spurious/override.yaml" \
	"WQM_CONFIG_PATH=${WORK}/spurious/config.yaml" \
	"WQM_MOUNTINFO_PATH=${WORK}/spurious/mountinfo-escaped"

echo "==> happy-path end-to-end (subtask 11.20)"
# Valid config + matching override + clean mountinfo → exit 0, no warnings.
cat >"${WORK}/spurious/mountinfo-clean" <<EOF
22 21 0:21 / /proc rw - proc proc rw
36 21 0:36 / ${WORK}/spurious/c1 rw - tmpfs tmpfs rw
37 21 0:37 / ${WORK}/spurious/c2 rw - tmpfs tmpfs rw
EOF
run_test "happy path passes all three layers" 0 \
	"layer 3: ok (no unexpected bind mounts)" \
	"WARNING" \
	"WQM_OVERRIDE_PATH=${WORK}/spurious/override.yaml" \
	"WQM_CONFIG_PATH=${WORK}/spurious/config.yaml" \
	"WQM_MOUNTINFO_PATH=${WORK}/spurious/mountinfo-clean"

# Absent override → layer 1 is skipped (the deployment hand-wires its own
# volumes; there is no generated override to go stale). Layers 2/3 still run:
# the spurious config declares c1/c2, both present, and mountinfo is clean, so
# the run exits 0. The hash check only guards a *mounted* override.
run_test "absent override skips layer 1 and continues" 0 \
	"layer 1: skipped (no override mounted" \
	"override file not found" \
	"WQM_OVERRIDE_PATH=${WORK}/spurious/no-such-override.yaml" \
	"WQM_CONFIG_PATH=${WORK}/spurious/config.yaml" \
	"WQM_MOUNTINFO_PATH=${WORK}/spurious/mountinfo-clean"

# Override without the hash header → layer 1 aborts with EXIT_HASH_MISMATCH.
cat >"${WORK}/spurious/override-no-header.yaml" <<'EOF'
services:
  memexd: {}
EOF
run_test "override missing hash header aborts layer 1" "${EXIT_HASH_MISMATCH}" \
	"missing the '# wqm-config-hash:" \
	"" \
	"WQM_OVERRIDE_PATH=${WORK}/spurious/override-no-header.yaml" \
	"WQM_CONFIG_PATH=${WORK}/spurious/config.yaml" \
	"WQM_MOUNTINFO_PATH=${WORK}/spurious/mountinfo-clean"

# Empty mounts list → all layers no-op cleanly.
mkdir -p "${WORK}/empty"
cat >"${WORK}/empty/config.yaml" <<'EOF'
mounts: []
EOF
HASH_E=$(spec_hash)
cat >"${WORK}/empty/override.yaml" <<EOF
# wqm-config-hash: ${HASH_E}
services:
  memexd: {}
EOF
run_test "empty mounts: every layer is a no-op" 0 \
	"layer 2: ok (no mounts declared in config.yaml)" \
	"" \
	"WQM_OVERRIDE_PATH=${WORK}/empty/override.yaml" \
	"WQM_CONFIG_PATH=${WORK}/empty/config.yaml" \
	"WQM_MOUNTINFO_PATH=${WORK}/nonexistent"

echo
printf 'Summary: %d passed, %d failed\n' "${PASS}" "${FAIL}"
if [ "${FAIL}" -gt 0 ]; then
	printf 'Failed scenarios:\n'
	for f in "${FAILURES[@]}"; do
		printf '  - %s\n' "${f}"
	done
	exit 1
fi
exit 0
