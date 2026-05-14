#!/usr/bin/env bash
# forbid_canonicalize.sh — CI guard preventing new canonicalize() calls without markers.
#
# Fails CI on any std::fs::canonicalize() or .canonicalize() call that lacks
# a // CATEGORY-B: marker on the same line or within 3 lines above.
#
# Phase A behavior (T5 execution, before T6 lands):
#   --baseline mode captures current violations to a snapshot file.
#   Without --baseline, compares against the snapshot and fails only on NEW violations.
#
# After T6 removes all Category A sites, the snapshot will be empty and the script
# will enforce zero new canonicalize() calls without markers.
#
# Usage:
#   ./scripts/ci/forbid_canonicalize.sh [--baseline] [<project_root>]
#
# Exit codes:
#   0 — no new violations (or baseline mode succeeded)
#   1 — new canonicalize() calls without markers found
#
# See docs/specs/16-path-abstraction.md §3.2.2 for Category B discipline.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${1:-.}"
BASELINE_MODE=0

# Handle --baseline flag
if [[ "${1:-}" == "--baseline" ]]; then
	BASELINE_MODE=1
	ROOT="${2:-.}"
fi

SNAPSHOT_FILE="$SCRIPT_DIR/forbid_canonicalize_baseline.txt"

echo "=== Forbid Canonicalize Check ==="
echo "Root: $ROOT"
if [[ $BASELINE_MODE -eq 1 ]]; then
	echo "Mode: BASELINE (snapshot current violations)"
else
	echo "Mode: ENFORCE (fail on new violations)"
fi
echo ""

# Find all canonicalize calls and check for CATEGORY-B marker
# Match patterns:
#   std::fs::canonicalize(
#   .canonicalize()

TEMP_VIOLATIONS="$SCRIPT_DIR/.forbid_canonicalize_temp.txt"
>"$TEMP_VIOLATIONS"

while IFS= read -r filepath; do
	[[ -z "$filepath" ]] && continue

	# Process each file to find canonicalize calls
	# Use awk to track line numbers and check for markers
	awk -v file="$filepath" '
    BEGIN {
        line_num = 0
        # Store last 3 lines for marker check
        for (i = 0; i < 3; i++) {
            prev_lines[i] = ""
        }
    }
    {
        line_num++
        # Shift previous lines
        for (i = 2; i >= 0; i--) {
            prev_lines[i+1] = prev_lines[i]
        }
        prev_lines[0] = $0

        # Check for canonicalize call patterns
        if ($0 ~ /std::fs::canonicalize\(/ || $0 ~ /\.canonicalize\(\)/) {
            # Check if CATEGORY-B marker is on this line or within 3 lines above
            has_marker = 0

            # Check current line
            if ($0 ~ /\/\/\s*CATEGORY-B:/) {
                has_marker = 1
            }

            # Check up to 3 lines above
            if (!has_marker) {
                for (i = 0; i <= 2; i++) {
                    if (prev_lines[i] ~ /\/\/\s*CATEGORY-B:/) {
                        has_marker = 1
                        break
                    }
                }
            }

            if (!has_marker) {
                print file ":" line_num ": " $0
            }
        }
    }
    ' "$filepath"

done < <(find "$ROOT/src/rust" -name "*.rs" -type f 2>/dev/null) >>"$TEMP_VIOLATIONS"

# Count violations
VIOLATION_COUNT=$(wc -l <"$TEMP_VIOLATIONS" | xargs)

if [[ $BASELINE_MODE -eq 1 ]]; then
	# Snapshot current violations
	cp "$TEMP_VIOLATIONS" "$SNAPSHOT_FILE"
	echo "Baseline snapshot created: $SNAPSHOT_FILE"
	echo "Current violation count: $VIOLATION_COUNT"
	echo ""

	if [[ $VIOLATION_COUNT -eq 0 ]]; then
		echo "✓ No canonicalize() calls without CATEGORY-B markers."
		rm "$TEMP_VIOLATIONS"
		exit 0
	else
		echo "ℹ Baseline captured. These Category A sites will be removed in T6."
		head -20 "$SNAPSHOT_FILE"
		if [[ $VIOLATION_COUNT -gt 20 ]]; then
			echo "... and $(($VIOLATION_COUNT - 20)) more"
		fi
		rm "$TEMP_VIOLATIONS"
		exit 0
	fi
else
	# Enforce: compare against snapshot
	if [[ ! -f "$SNAPSHOT_FILE" ]]; then
		echo "ERROR: Baseline snapshot not found: $SNAPSHOT_FILE" >&2
		echo "Run with --baseline flag first to create snapshot." >&2
		rm "$TEMP_VIOLATIONS"
		exit 1
	fi

	# Find new violations not in baseline
	BASELINE_COUNT=$(wc -l <"$SNAPSHOT_FILE" | xargs)
	NEW_VIOLATIONS_COUNT=$(comm -13 <(sort "$SNAPSHOT_FILE" | cut -d: -f1-2) <(sort "$TEMP_VIOLATIONS" | cut -d: -f1-2) | wc -l | xargs)

	if [[ $NEW_VIOLATIONS_COUNT -eq 0 ]]; then
		echo "✓ No new canonicalize() calls without CATEGORY-B markers."
		echo "  (Baseline contains $BASELINE_COUNT Category A sites; will be removed in T6)"
		rm "$TEMP_VIOLATIONS"
		exit 0
	else
		echo "✗ Found $NEW_VIOLATIONS_COUNT new canonicalize() call(s) without CATEGORY-B marker:" >&2
		comm -13 <(sort "$SNAPSHOT_FILE") <(sort "$TEMP_VIOLATIONS") >&2
		echo "" >&2
		echo "Add a // CATEGORY-B: marker if this site is safe (process-local use only)." >&2
		rm "$TEMP_VIOLATIONS"
		exit 1
	fi
fi
