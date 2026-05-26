#!/bin/sh
# run-sh.sh — exercise scripts/lib/path-resolver.sh against cases.json.
#
# Validates the predicate helpers (`is_windows_absolute`, `is_absolute`)
# and the `normalize_slashes` transform. The full canonical-path
# normalization in this fixture is not currently re-implemented in shell
# — shell consumers only need the simpler helpers. The Rust and (future)
# TS runners cover the `normalize` / `normalize_errors` sections.
#
# Usage:
#   tests/path-fixtures/run-sh.sh
#
# Required: sh, jq

set -u

SELF_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SELF_DIR/../.." && pwd)"
FIXTURES="$SELF_DIR/cases.json"
LIB="$REPO_ROOT/scripts/lib/path-resolver.sh"

PY=""
for cand in python3 python py; do
  if command -v "$cand" >/dev/null 2>&1; then
    PY="$cand"; break
  fi
done
[ -n "$PY" ] || {
  printf 'python (any of python3/python/py) is required to parse the fixture JSON.\n' >&2
  exit 2
}

# Reads one section into "<input>\t<expected>\n..." lines via stdout.
# Boolean expecteds become the strings "true"/"false" so we can compare
# against the shell helpers' string output.
emit_cases() {
  _section="$1"
  _input_key="${2:-input}"
  _expected_key="${3:-expected}"
  # Force LF line endings even on Windows Python.
  "$PY" - "$FIXTURES" "$_section" "$_input_key" "$_expected_key" <<'PY_END'
import json, sys
fixtures, section, input_key, expected_key = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
with open(fixtures, encoding='utf-8') as fh:
    data = json.load(fh)
out_lines = []
for case in data[section]:
    inp = case[input_key]
    exp = case[expected_key]
    if isinstance(exp, bool):
        exp = "true" if exp else "false"
    # Two-line encoding: input on one line, expected on the next.
    out_lines.append(f"{inp}\n{exp}\n")
sys.stdout.buffer.write("".join(out_lines).encode("utf-8"))
PY_END
}

# shellcheck source=../../scripts/lib/path-resolver.sh
. "$LIB"

PASS=0
FAIL=0
FAILED_CASES=""

record_fail() {
  FAIL=$((FAIL + 1))
  FAILED_CASES="${FAILED_CASES}
  $1"
}

TMP="$(mktemp 2>/dev/null || printf '/tmp/wqm-fixtures-%d' "$$")"
trap 'rm -f "$TMP"' EXIT INT TERM

# Run each section by emitting its cases as two lines per case
# (input on one line, expected on the next). Two-line encoding avoids
# POSIX shell `read` quirks with empty leading fields when splitting
# on a separator.
run_section() {
  _label="$1"
  _fn="$2"
  emit_cases "$_label" > "$TMP"
  while IFS= read -r input && IFS= read -r expected; do
    actual=$($_fn "$input")
    if [ "$actual" = "$expected" ]; then
      PASS=$((PASS + 1))
    else
      record_fail "$_label input=$input expected=$expected actual=$actual"
    fi
  done < "$TMP"
}

run_section normalize_slashes   wqm_path_normalize_slashes
run_section is_windows_absolute wqm_path_is_windows_absolute
run_section is_absolute         wqm_path_is_absolute

printf 'sh path-fixtures: %d passed, %d failed\n' "$PASS" "$FAIL"
if [ "$FAIL" -gt 0 ]; then
  # `%s` (not raw interpolation) — strings contain backslashes that
  # printf would otherwise interpret as escape sequences (\f, \U, ...).
  printf '%s\n' "$FAILED_CASES"
  exit 1
fi
exit 0
