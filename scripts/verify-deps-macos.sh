#!/usr/bin/env bash
# Verify macOS binaries have only allowed dynamic dependencies.
# Allows system libraries in /usr/lib/ and /System/Library/Frameworks/.
# Fails with exit code 1 if unexpected libraries are found (e.g., homebrew).
set -euo pipefail

check_binary() {
    local binary="$1"
    if [[ ! -f "$binary" ]]; then
        echo "ERROR: Binary not found: $binary"
        return 1
    fi

    echo "Checking dependencies for: $binary"
    local deps
    deps=$(otool -L "$binary" 2>&1)

    local violations=0
    local first_line=true
    while IFS= read -r line; do
        # Skip first line (binary name header)
        if [[ "$first_line" == "true" ]]; then
            first_line=false
            continue
        fi

        # Skip empty lines
        [[ -z "$line" ]] && continue

        # Extract library path (otool format: "\tpath (compatibility version X, current version Y)")
        local lib_path
        lib_path=$(echo "$line" | awk '{print $1}')

        if [[ "$lib_path" == /usr/lib/* ]] || [[ "$lib_path" == /System/Library/* ]]; then
            echo "  OK: $lib_path"
        elif [[ "$lib_path" == @rpath/* ]]; then
            # @rpath references are typically framework-internal, check they're not external
            echo "  WARN: @rpath reference: $lib_path (verify this is framework-internal)"
        else
            echo "  VIOLATION: Unexpected dependency: $lib_path"
            ((violations++)) || true
        fi
    done <<< "$deps"

    if [[ $violations -gt 0 ]]; then
        echo "FAILED: $violations unexpected dependencies found in $binary"
        return 1
    fi

    echo "PASSED: All dependencies are allowed for $binary"
    return 0
}

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 <binary> [binary...]"
    exit 1
fi

exit_code=0
for binary in "$@"; do
    if ! check_binary "$binary"; then
        exit_code=1
    fi
    echo ""
done

exit $exit_code
