#!/usr/bin/env bash
# Run every wqm-0.2 CI guard, report each, and fail if any fails. Callable locally
# (`ci/run_guards.sh` from src/rust) and from the CI workflow. Each guard prints its
# own PASS/FAIL line -- including the honest "PASS (vacuous)" lines on the Phase-0
# skeleton, so a green run never hides that a guard had nothing to check yet.
set -uo pipefail

cd "$(dirname "$0")/.." || exit 2
here="ci"

guards=(
	"guard_link_closure.py"           # N14 link-closure, all 4 directions (default-deny)
	"guard_stratification.py"         # §9.1 stratification + whole-workspace acyclicity
	"guard_codesize.py"               # §9 limits: 500 lines/file, 80 lines/function
	"guard_name_registry.py"          # N8 name-registry single producer
	"guard_field_family.py"           # N35 field-family completeness
	"guard_dedash_single_producer.py" # N3 fts_key single producer
	"guard_mac_single_setter.py"      # N5 MAC-marker single setter
	"guard_no_skipped_tests.py"       # TDD charter: no disabled tests
	"guard_bin_suffix.py"             # -v2 parallel deployment: bin filenames differ

	# Selftests: prove a guard FAILS where it claims to, against throwaway
	# fixtures. A guard that has never failed is indistinguishable from one that
	# cannot fail -- and most guards above are still vacuous on this workspace,
	# so their PASS lines rest entirely on these (P04-GT001-WO005).
	"selftest_link_closure.py"     # 14 cases over the N14 guard's 4 directions
	"selftest_stratification.py"   # 10 cases -- up-edges, same-level edges, exemption
	"selftest_codesize.py"         # 9 cases -- both limits, and the literal stripping
	"selftest_no_skipped_tests.py" # 9 cases -- and that PROSE about the ban is legal
	"selftest_name_registry.py"    # 12 cases -- and that PROSE quoting a name is legal
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
