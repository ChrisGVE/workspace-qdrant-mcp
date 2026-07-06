#!/usr/bin/env bash
# Codesize report (coding.md §VIII limits: Rust 500 lines/file). Non-gating in CI --
# it reports so oversized files are visible and split deliberately, per the
# "refactor gradually by extracting the section being changed" rule. Uses tokei
# when available for a summary, then lists any *.rs file over the line limit.
set -uo pipefail

cd "$(dirname "$0")/.." || exit 2
LIMIT=500

echo "== codesize summary =="
if command -v tokei >/dev/null 2>&1; then
	tokei --type Rust . 2>/dev/null || tokei .
else
	echo "(tokei not installed -- skipping language summary)"
fi

echo
echo "== Rust files over ${LIMIT} lines =="
over=0
while IFS= read -r -d '' f; do
	n=$(wc -l <"$f")
	if [ "$n" -gt "$LIMIT" ]; then
		printf '  %6d  %s\n' "$n" "${f#./}"
		over=$((over + 1))
	fi
done < <(find . -path ./target -prune -o -name '*.rs' -print0)

if [ "$over" -eq 0 ]; then
	echo "  none -- all Rust files within the ${LIMIT}-line limit."
fi
echo
echo "codesize: report only (non-gating)."
