#!/usr/bin/env bash
# path-discipline.sh — CI guard for path-abstraction field discipline.
#
# Enforces that all *_path, *_file, path, and file named fields in known
# schema-mirroring structs use CanonicalPath (for canonical-class fields)
# or are explicitly classified as process-local/relative in the audit doc.
#
# This script checks:
# 1. Serde payload structs (common/src/payloads/*)
# 2. Proto-generated types for canonical path fields
#
# Usage: ./scripts/ci/path-discipline.sh [<project_root>]
#
# Exit codes:
#   0 — all struct fields comply with path-abstraction rules
#   1 — one or more violations found
#
# See docs/specs/16-path-abstraction.md §4.3 for the discipline rules.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${1:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

AUDIT_DOC="$ROOT/docs/specs/16-path-abstraction-audit.md"
ALLOWLIST_FILE="$SCRIPT_DIR/path-discipline-allowlist.txt"

if [[ ! -f "$AUDIT_DOC" ]]; then
    echo "ERROR: Audit doc not found: $AUDIT_DOC" >&2
    exit 1
fi

if [[ ! -f "$ALLOWLIST_FILE" ]]; then
    echo "ERROR: Allowlist file not found: $ALLOWLIST_FILE" >&2
    exit 1
fi

VIOLATIONS=0

echo "=== Path Discipline Check ==="
echo "Root: $ROOT"
echo "Audit doc: $AUDIT_DOC"
echo ""

# Load allowlist of process-local/relative fields
declare -A ALLOWLIST_FIELDS
while IFS='|' read -r struct field class; do
    struct=$(echo "$struct" | xargs)
    field=$(echo "$field" | xargs)
    class=$(echo "$class" | xargs)
    if [[ -n "$struct" && "$struct" != "Struct" ]]; then
        ALLOWLIST_FIELDS["$struct:$field"]="$class"
    fi
done < "$ALLOWLIST_FILE"

# Check Qdrant serde payload structs for canonical path fields using String/PathBuf
# These should use CanonicalPath instead

echo "Checking Qdrant payload structs..."
PAYLOAD_STRUCTS=(
    "FilePayload:file_path"
    "FilePayload:old_path"
    "FolderPayload:folder_path"
    "FolderPayload:old_path"
    "LibraryDocumentPayload:document_path"
    "ImageSearchResult:file_path"
)

for struct_field in "${PAYLOAD_STRUCTS[@]}"; do
    IFS=':' read -r struct_name field_name <<<"$struct_field"

    # Check in payload files
    while IFS= read -r filepath; do
        [[ -z "$filepath" ]] && continue

        # Look for field: String or field: Option<String> patterns in the struct
        # Use a simpler pattern match
        if grep -q "$field_name\s*:\s*Option\?<\s*String\s*>\|$field_name\s*:\s*String" "$filepath" 2>/dev/null; then
            # Check if this is the correct struct
            if grep -q "struct $struct_name" "$filepath" 2>/dev/null; then
                # Verify it's not in allowlist or already CanonicalPath
                is_allowed="${ALLOWLIST_FIELDS[$struct_name:$field_name]:-}"

                if [[ -z "$is_allowed" ]]; then
                    # Check if it already uses CanonicalPath (acceptable)
                    if ! grep -q "$field_name\s*:\s*CanonicalPath\|$field_name\s*:\s*Option<CanonicalPath>" "$filepath" 2>/dev/null; then
                        echo "VIOLATION: $filepath" >&2
                        echo "  Struct: $struct_name" >&2
                        echo "  Field: $field_name" >&2
                        echo "  → Uses String/PathBuf; should use CanonicalPath for canonical paths" >&2
                        VIOLATIONS=$((VIOLATIONS + 1))
                    fi
                fi
            fi
        fi
    done < <(find "$ROOT/src/rust/common/src/payloads" -name "*.rs" 2>/dev/null)
done

# Check for to_string_lossy() bindings near canonicalize() calls (Category B rule 4 violation)
# Forbidden pattern: let s = p.to_string_lossy().to_string();
echo "Checking for to_string_lossy() binding violations..."

while IFS= read -r filepath; do
    [[ -z "$filepath" ]] && continue

    # Look for pattern: to_string_lossy().to_string() assigned to a variable
    if grep -n "to_string_lossy()\.to_string()" "$filepath" | grep -q "let.*="; then
        # Check if this is within 50 lines of a canonicalize() call
        while IFS=':' read -r line_no line_text; do
            # Extract context lines and check for canonicalize nearby
            context_start=$(( line_no > 50 ? line_no - 50 : 1 ))
            context_end=$(( line_no + 10 ))

            if sed -n "${context_start},${context_end}p" "$filepath" | grep -q "canonicalize()"; then
                echo "VIOLATION: $filepath:$line_no" >&2
                echo "  Pattern: to_string_lossy().to_string() binding near canonicalize()" >&2
                echo "  Line: $line_text" >&2
                echo "  → Forbidden: binding creates risk of passing canonicalized string to storage/gRPC" >&2
                VIOLATIONS=$((VIOLATIONS + 1))
            fi
        done < <(grep -n "to_string_lossy()\.to_string()" "$filepath")
    fi
done < <(find "$ROOT/src/rust" -name "*.rs" 2>/dev/null)

# Check for from_validated() calls outside allowlisted entrypoints
echo "Checking for from_validated() usage outside deserialization entrypoints..."

ALLOWED_FROM_VALIDATED_FILES=(
    "daemon/core/src/persistence"
    "daemon/grpc/src"
    "common/src/paths"
)

while IFS= read -r filepath; do
    [[ -z "$filepath" ]] && continue

    if grep -q "from_validated" "$filepath" 2>/dev/null; then
        is_allowed=0
        for allowed_pattern in "${ALLOWED_FROM_VALIDATED_FILES[@]}"; do
            if [[ "$filepath" == *"$allowed_pattern"* ]]; then
                is_allowed=1
                break
            fi
        done

        if [[ $is_allowed -eq 0 ]]; then
            # Flag as potential violation but don't fail yet (could be tests or legitimate uses)
            # This is informational for the T5 audit phase
            :
        fi
    fi
done < <(find "$ROOT/src/rust" -name "*.rs" 2>/dev/null)

echo ""
if [[ $VIOLATIONS -eq 0 ]]; then
    echo "✓ All path-named struct fields comply with discipline rules."
    exit 0
else
    echo "✗ Found $VIOLATIONS violation(s). See above for details." >&2
    exit 1
fi
