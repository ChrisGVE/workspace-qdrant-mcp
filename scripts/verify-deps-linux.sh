#!/usr/bin/env bash
# Verify Linux binaries have only allowed dynamic dependencies.
# Fails with exit code 1 if unexpected libraries are found.
set -euo pipefail

ALLOWED_LIBS=(
    "linux-vdso.so"
    "ld-linux"
    "libc.so"
    "libm.so"
    "libdl.so"
    "libpthread.so"
    "libgcc_s.so"
    "libstdc++.so"    # Required by ONNX Runtime C++ runtime
    "librt.so"
)

check_binary() {
    local binary="$1"
    if [[ ! -f "$binary" ]]; then
        echo "ERROR: Binary not found: $binary"
        return 1
    fi

    echo "Checking dependencies for: $binary"

    # Try ldd first; fall back to readelf for cross-compiled binaries
    local deps
    local use_readelf=false
    deps=$(ldd "$binary" 2>&1) || true

    if echo "$deps" | grep -qE "not a dynamic executable|cannot execute"; then
        # Try readelf for cross-compiled or statically linked binaries
        if command -v readelf &>/dev/null; then
            local needed
            needed=$(readelf -d "$binary" 2>/dev/null | grep NEEDED || true)
            if [[ -z "$needed" ]]; then
                echo "  OK: Statically linked (no dynamic dependencies)"
                return 0
            fi
            deps="$needed"
            use_readelf=true
        else
            echo "  OK: Statically linked (no dynamic dependencies)"
            return 0
        fi
    fi

    local violations=0
    while IFS= read -r line; do
        [[ -z "$line" ]] && continue

        local lib_name
        if [[ "$use_readelf" == "true" ]]; then
            # readelf format: " 0x... (NEEDED) Shared library: [libc.so.6]"
            lib_name=$(echo "$line" | grep -oP '\[\K[^\]]+')
        else
            # ldd format: "libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6"
            lib_name=$(echo "$line" | awk '{print $1}')
        fi

        [[ -z "$lib_name" ]] && continue

        local allowed=false
        for pattern in "${ALLOWED_LIBS[@]}"; do
            if [[ "$lib_name" == *"$pattern"* ]]; then
                allowed=true
                break
            fi
        done

        if [[ "$allowed" == "false" ]]; then
            echo "  VIOLATION: Unexpected dependency: $lib_name"
            ((violations++)) || true
        else
            echo "  OK: $lib_name"
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
