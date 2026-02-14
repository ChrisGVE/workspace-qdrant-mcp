#!/usr/bin/env bash
#
# validate.sh — Validate test corpus file classification
#
# This script verifies that the daemon correctly classifies and processes
# each test corpus file. It checks the tracked_files table in SQLite for
# correct file_type, language, extension, and is_test metadata.
#
# Prerequisites:
#   - The test-corpus/ directory must be registered as a project with wqm
#   - The daemon must have finished processing (queue drained)
#
# Usage:
#   ./validate.sh [--register] [--wait] [--verbose]
#
#   --register   Register test-corpus/ as a project first
#   --wait       Wait for queue to drain before validating
#   --verbose    Show details for passing checks too

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DB_PATH="${HOME}/.workspace-qdrant/state.db"
PASS=0
FAIL=0
SKIP=0
VERBOSE=false
DO_REGISTER=false
DO_WAIT=false

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BOLD='\033[1m'
NC='\033[0m'

for arg in "$@"; do
    case "$arg" in
        --register)  DO_REGISTER=true ;;
        --wait)      DO_WAIT=true ;;
        --verbose)   VERBOSE=true ;;
        *)           echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

# --- Helpers ---

log_pass() {
    PASS=$((PASS + 1))
    if $VERBOSE; then
        echo -e "  ${GREEN}PASS${NC} $1"
    fi
}

log_fail() {
    FAIL=$((FAIL + 1))
    echo -e "  ${RED}FAIL${NC} $1"
    echo -e "       expected: $2"
    echo -e "       got:      $3"
}

log_skip() {
    SKIP=$((SKIP + 1))
    echo -e "  ${YELLOW}SKIP${NC} $1"
}

# Query tracked_files for a specific file path
query_tracked() {
    local file_path="$1"
    local column="$2"
    sqlite3 "$DB_PATH" "SELECT $column FROM tracked_files WHERE file_path = '$file_path' LIMIT 1;" 2>/dev/null || echo ""
}

# Check if a file exists in tracked_files
file_tracked() {
    local file_path="$1"
    local count
    count=$(sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM tracked_files WHERE file_path = '$file_path';" 2>/dev/null || echo "0")
    [ "$count" -gt 0 ]
}

# Validate a single file's classification
check_file() {
    local rel_path="$1"        # Relative to test-corpus/
    local exp_file_type="$2"
    local exp_language="$3"
    local exp_extension="$4"
    local exp_is_test="$5"     # 0 or 1

    local full_path="${SCRIPT_DIR}/${rel_path}"
    local label="${rel_path}"

    if [ ! -f "$full_path" ]; then
        log_skip "$label (file does not exist)"
        return
    fi

    # Find the file in tracked_files by matching the tail of file_path
    if ! file_tracked "$full_path"; then
        log_skip "$label (not in tracked_files — may not be ingested yet)"
        return
    fi

    echo -e "${BOLD}$label${NC}"

    # Check file_type
    local actual_ft
    actual_ft=$(query_tracked "$full_path" "file_type")
    if [ "$actual_ft" = "$exp_file_type" ]; then
        log_pass "file_type=$actual_ft"
    else
        log_fail "file_type" "$exp_file_type" "$actual_ft"
    fi

    # Check language
    local actual_lang
    actual_lang=$(query_tracked "$full_path" "language")
    if [ "$actual_lang" = "$exp_language" ]; then
        log_pass "language=$actual_lang"
    else
        log_fail "language" "$exp_language" "$actual_lang"
    fi

    # Check extension
    local actual_ext
    actual_ext=$(query_tracked "$full_path" "extension")
    if [ "$actual_ext" = "$exp_extension" ]; then
        log_pass "extension=$actual_ext"
    else
        log_fail "extension" "$exp_extension" "$actual_ext"
    fi

    # Check is_test
    local actual_test
    actual_test=$(query_tracked "$full_path" "is_test")
    if [ "$actual_test" = "$exp_is_test" ]; then
        log_pass "is_test=$actual_test"
    else
        log_fail "is_test" "$exp_is_test" "$actual_test"
    fi
}

# --- Pre-checks ---

if [ ! -f "$DB_PATH" ]; then
    echo "Error: SQLite database not found at $DB_PATH"
    echo "Is the daemon running?"
    exit 1
fi

if $DO_REGISTER; then
    echo "Registering test-corpus/ as a project..."
    wqm project register "$SCRIPT_DIR" --name "test-corpus" -y
    echo ""
fi

if $DO_WAIT; then
    echo "Waiting for queue to drain..."
    max_wait=120
    elapsed=0
    while [ $elapsed -lt $max_wait ]; do
        pending=$(sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM unified_queue WHERE status IN ('pending', 'in_progress');" 2>/dev/null || echo "0")
        if [ "$pending" = "0" ]; then
            echo "Queue empty."
            break
        fi
        echo "  $pending items remaining..."
        sleep 5
        elapsed=$((elapsed + 5))
    done
    if [ $elapsed -ge $max_wait ]; then
        echo "Warning: queue not fully drained after ${max_wait}s, proceeding anyway."
    fi
    echo ""
fi

# --- Validation ---

echo -e "${BOLD}=== Test Corpus Validation ===${NC}"
echo ""

# --- Text files ---
echo -e "${BOLD}--- Text ---${NC}"
#                   rel_path                    file_type  language    extension  is_test
check_file "text/plain.txt"                     "text"     ""          "txt"      "0"
check_file "text/unicode.txt"                   "text"     ""          "txt"      "0"
check_file "text/markdown_complex.md"           "text"     ""          "md"       "0"
echo ""

# --- Documents ---
echo -e "${BOLD}--- Documents ---${NC}"
check_file "docs/simple.pdf"                    "docs"     ""          "pdf"      "0"
check_file "docs/images.pdf"                    "docs"     ""          "pdf"      "0"
check_file "docs/fonts.pdf"                     "docs"     ""          "pdf"      "0"
check_file "docs/sample.docx"                   "docs"     ""          "docx"     "0"
check_file "docs/formatted.docx"                "docs"     ""          "docx"     "0"
check_file "docs/sample.odt"                    "docs"     ""          "odt"      "0"
check_file "docs/sample.rtf"                    "docs"     ""          "rtf"      "0"
echo ""

# --- Slides ---
echo -e "${BOLD}--- Slides ---${NC}"
check_file "slides/sample.pptx"                 "slides"   ""          "pptx"     "0"
check_file "slides/sample.odp"                  "slides"   ""          "odp"      "0"
echo ""

# --- Ebooks ---
echo -e "${BOLD}--- Ebooks ---${NC}"
check_file "ebooks/sample.epub"                 "docs"     ""          "epub"     "0"
echo ""

# --- Web ---
echo -e "${BOLD}--- Web ---${NC}"
check_file "web/simple.html"                    "web"      "html"      "html"     "0"
check_file "web/complex.html"                   "web"      "html"      "html"     "0"
check_file "web/semantic.html"                  "web"      "html"      "html"     "0"
check_file "web/styles.css"                     "web"      "css"       "css"      "0"
echo ""

# --- Website ---
echo -e "${BOLD}--- Website ---${NC}"
check_file "website/index.html"                 "web"      "html"      "html"     "0"
check_file "website/about.html"                 "web"      "html"      "html"     "0"
check_file "website/contact.html"               "web"      "html"      "html"     "0"
check_file "website/style.css"                  "web"      "css"       "css"      "0"
echo ""

# --- Code ---
echo -e "${BOLD}--- Code ---${NC}"
check_file "code/sample.ps1"                    "code"     "powershell"  "ps1"    "0"
check_file "code/sample.d"                      "code"     "d"           "d"      "0"
check_file "code/types.d.ts"                    "code"     "typescript"  "d.ts"   "0"
check_file "code/test_example.py"               "code"     "python"      "py"     "1"
check_file "code/utils.py"                      "code"     "python"      "py"     "0"
echo ""

# --- Config ---
echo -e "${BOLD}--- Config ---${NC}"
check_file "config/settings.yaml"               "config"   "yaml"       "yaml"    "0"
check_file "config/settings.toml"               "config"   "toml"       "toml"    "0"
echo ""

# --- Data ---
echo -e "${BOLD}--- Data ---${NC}"
check_file "data/sample.json"                   "data"     "json"       "json"    "0"
echo ""

# --- Summary ---
echo -e "${BOLD}=== Summary ===${NC}"
TOTAL=$((PASS + FAIL + SKIP))
echo -e "  ${GREEN}Passed${NC}: $PASS"
echo -e "  ${RED}Failed${NC}: $FAIL"
echo -e "  ${YELLOW}Skipped${NC}: $SKIP"
echo "  Total checks: $TOTAL"
echo ""

if [ "$FAIL" -gt 0 ]; then
    echo -e "${RED}VALIDATION FAILED${NC}"
    exit 1
else
    echo -e "${GREEN}ALL CHECKS PASSED${NC}"
    exit 0
fi
