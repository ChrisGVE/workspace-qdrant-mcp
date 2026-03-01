#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/models.sh"
source "${SCRIPT_DIR}/storage.sh"
source "${SCRIPT_DIR}/utils.sh"

main() {
    init_shelf "Computer Science" 10

    add_book "$(make_book "The Art of Computer Programming" "Donald Knuth" 1968 "9780201896831" "true")"
    add_book "$(make_book "Structure and Interpretation of Computer Programs" "Harold Abelson" 1996 "9780262510875" "true")"
    add_book "$(make_book "Introduction to Algorithms" "Thomas Cormen" 2009 "9780262033848" "false")"
    add_book "$(make_book "Design Patterns" "Erich Gamma" 1994 "9780201633610" "true")"
    add_book "$(make_book "The Pragmatic Programmer" "David Thomas" 2019 "9780135957059" "true")"

    generate_report
    echo ""

    echo '--- Search by author "knuth" ---'
    while IFS= read -r book; do
        [ -z "$book" ] && continue
        echo "  $(format_book "$book")"
    done <<< "$(find_by_author "knuth")"
    echo ""

    echo "--- Search by year range 1990-2010 ---"
    while IFS= read -r book; do
        [ -z "$book" ] && continue
        echo "  $(format_book "$book")"
    done <<< "$(find_by_year_range 1990 2010)"
    echo ""

    echo "--- Parse CSV ---"
    local parsed
    parsed=$(parse_csv_line "Clean Code,Robert Martin,2008,9780132350884,true")
    echo "  Parsed: $(format_book "$parsed")"
    echo ""

    echo "--- ISBN Validation ---"
    for book in "${SHELF_BOOKS[@]}"; do
        local isbn status
        isbn=$(book_isbn "$book")
        status=$(validate_isbn "$isbn")
        if [ "$status" = "true" ]; then
            echo "  ${isbn}: valid"
        else
            echo "  ${isbn}: invalid"
        fi
    done
}

main
