#!/usr/bin/env bash
# check-version-consistency.sh
#
# Validates version consistency across configuration files and CI workflows.
# Run from the repository root.
#
# Checks:
#   1. ORT_VERSION is identical across all workflow files that use it
#   2. default_configuration.yaml tree_sitter_version matches Cargo.lock
#   3. Cargo.lock contains tree-sitter (sanity check for build.rs)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXIT_CODE=0

echo "=== Version Consistency Check ==="
echo ""

# ── Check 1: ORT_VERSION consistency across workflows ──────────────────

echo "--- ORT_VERSION consistency ---"

ORT_VERSIONS=()
while IFS= read -r line; do
    file=$(echo "$line" | cut -d: -f1)
    version=$(echo "$line" | grep -oE '"[0-9]+\.[0-9]+\.[0-9]+"' | tr -d '"')
    if [ -n "$version" ]; then
        ORT_VERSIONS+=("$file:$version")
        echo "  $file: $version"
    fi
done < <(grep -rn 'ORT_VERSION:' "$REPO_ROOT/.github/workflows/" 2>/dev/null || true)

if [ ${#ORT_VERSIONS[@]} -eq 0 ]; then
    echo "  WARNING: No ORT_VERSION found in any workflow file"
else
    # Extract unique versions
    UNIQUE_VERSIONS=$(printf '%s\n' "${ORT_VERSIONS[@]}" | sed 's/.*://' | sort -u)
    COUNT=$(echo "$UNIQUE_VERSIONS" | wc -l | tr -d ' ')

    if [ "$COUNT" -gt 1 ]; then
        echo ""
        echo "  ERROR: ORT_VERSION mismatch across workflow files!"
        echo "  Found versions: $(echo "$UNIQUE_VERSIONS" | tr '\n' ' ')"
        EXIT_CODE=1
    else
        echo "  OK: All workflow files use ORT_VERSION=$UNIQUE_VERSIONS"
    fi
fi
echo ""

# ── Check 2: tree_sitter_version in YAML config matches Cargo.lock ────

echo "--- Tree-sitter version (YAML config vs Cargo.lock) ---"

LOCK_FILE="$REPO_ROOT/src/rust/Cargo.lock"
YAML_FILE="$REPO_ROOT/assets/default_configuration.yaml"

if [ ! -f "$LOCK_FILE" ]; then
    echo "  WARNING: Cargo.lock not found at $LOCK_FILE"
else
    # Extract tree-sitter version from Cargo.lock
    # Format: name = "tree-sitter" followed by version = "X.Y.Z"
    TS_LOCK_VERSION=$(awk '/^name = "tree-sitter"$/{found=1; next} found && /^version =/{gsub(/"/, "", $3); print $3; exit}' "$LOCK_FILE")
    TS_LOCK_MAJOR_MINOR=$(echo "$TS_LOCK_VERSION" | cut -d. -f1,2)

    echo "  Cargo.lock tree-sitter version: $TS_LOCK_VERSION (major.minor: $TS_LOCK_MAJOR_MINOR)"

    if [ -f "$YAML_FILE" ]; then
        # Extract tree_sitter_version from YAML config (skip comment lines)
        TS_YAML_VERSION=$(grep -v '^ *#' "$YAML_FILE" | grep 'tree_sitter_version:' | head -1 | sed 's/.*: *"\([^"]*\)".*/\1/')
        echo "  default_configuration.yaml: $TS_YAML_VERSION"

        if [ "$TS_LOCK_MAJOR_MINOR" != "$TS_YAML_VERSION" ]; then
            echo ""
            echo "  ERROR: tree_sitter_version mismatch!"
            echo "  Cargo.lock major.minor: $TS_LOCK_MAJOR_MINOR"
            echo "  YAML config:            $TS_YAML_VERSION"
            echo "  Update assets/default_configuration.yaml to match."
            EXIT_CODE=1
        else
            echo "  OK: YAML config matches Cargo.lock"
        fi
    else
        echo "  WARNING: default_configuration.yaml not found at $YAML_FILE"
    fi
fi
echo ""

# ── Check 3: Cargo.lock sanity check for build.rs ─────────────────────

echo "--- Cargo.lock sanity check ---"

if [ -f "$LOCK_FILE" ]; then
    if grep -q '^name = "tree-sitter"$' "$LOCK_FILE"; then
        echo "  OK: tree-sitter package found in Cargo.lock"
    else
        echo "  ERROR: tree-sitter package NOT found in Cargo.lock"
        echo "  build.rs will emit TREE_SITTER_VERSION=unknown"
        EXIT_CODE=1
    fi
else
    echo "  WARNING: Cargo.lock not found"
fi
echo ""

# ── Summary ────────────────────────────────────────────────────────────

if [ $EXIT_CODE -eq 0 ]; then
    echo "=== All version consistency checks passed ==="
else
    echo "=== Version consistency checks FAILED ==="
fi

exit $EXIT_CODE
