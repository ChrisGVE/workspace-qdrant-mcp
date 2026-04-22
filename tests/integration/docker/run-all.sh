#!/usr/bin/env bash
# tests/integration/docker/run-all.sh
#
# Driver: run every test-*.sh in this directory sequentially, each in its
# own compose project so state doesn't leak. Intended for CI; runs fine
# locally as well if docker + docker compose are installed.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v docker >/dev/null 2>&1; then
	echo "docker is required" >&2
	exit 1
fi
if ! docker compose version >/dev/null 2>&1; then
	echo "docker compose v2 is required" >&2
	exit 1
fi

failed=()
for test in "${SCRIPT_DIR}"/test-*.sh; do
	name="$(basename "${test}")"
	echo ""
	echo "════════════════════════════════════════════════════════════════════════"
	echo " Running ${name}"
	echo "════════════════════════════════════════════════════════════════════════"
	if bash "${test}"; then
		echo "── ${name} PASSED ──"
	else
		echo "── ${name} FAILED ──"
		failed+=("${name}")
	fi
done

echo ""
if ((${#failed[@]} > 0)); then
	echo "${#failed[@]} test(s) failed: ${failed[*]}" >&2
	exit 1
fi
echo "All integration tests passed"
