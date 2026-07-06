#!/usr/bin/env bash
# Run every wqm-0.2 CI guard, report each, and fail if any fails. Callable locally
# (`ci/run_guards.sh` from src/rust) and from the CI workflow. Each guard prints its
# own PASS/FAIL line -- including the honest "PASS (vacuous)" lines on the Phase-0
# skeleton, so a green run never hides that a guard had nothing to check yet.
set -uo pipefail

cd "$(dirname "$0")/.." || exit 2
here="ci"

guards=(
	"guard_link_closure.py"           # N14 link-closure (default-deny)
	"guard_field_family.py"           # N35 field-family completeness
	"guard_dedash_single_producer.py" # N3 fts_key single producer
	"guard_mac_single_setter.py"      # N5 MAC-marker single setter
	"guard_no_skipped_tests.py"       # TDD charter: no disabled tests
)

failed=0
for g in "${guards[@]}"; do
	echo "== ${g} =="
	if ! python3 "${here}/${g}"; then
		failed=1
	fi
done

if [ "${failed}" -ne 0 ]; then
	echo "run_guards: FAIL -- one or more guards failed." >&2
	exit 1
fi
echo "run_guards: all guards passed."
