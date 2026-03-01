#!/usr/bin/env bash
# Storage: business logic for managing books on the shelf

add_book() {
    local book="$1"
    if is_full; then
        echo "Error: Shelf '${SHELF_NAME}' is at capacity (${SHELF_CAPACITY})" >&2
        return 1
    fi
    SHELF_BOOKS+=("$book")
}

remove_book() {
    local isbn="$1"
    local new_books=()
    local found=0
    for book in "${SHELF_BOOKS[@]}"; do
        if [ "$(book_isbn "$book")" = "$isbn" ] && [ "$found" -eq 0 ]; then
            found=1
        else
            new_books+=("$book")
        fi
    done
    if [ "$found" -eq 0 ]; then
        echo "Error: No book with ISBN '${isbn}' found on shelf" >&2
        return 1
    fi
    SHELF_BOOKS=("${new_books[@]}")
}

find_by_author() {
    local query
    query=$(echo "$1" | tr '[:upper:]' '[:lower:]')
    for book in "${SHELF_BOOKS[@]}"; do
        local author
        author=$(book_author "$book" | tr '[:upper:]' '[:lower:]')
        if [[ "$author" == *"$query"* ]]; then
            echo "$book"
        fi
    done
}

find_by_year_range() {
    local start_year="$1" end_year="$2"
    for book in "${SHELF_BOOKS[@]}"; do
        local year
        year=$(book_year "$book")
        if [ "$year" -ge "$start_year" ] && [ "$year" -le "$end_year" ]; then
            echo "$book"
        fi
    done
}

sort_by_title() {
    local tmpfile
    tmpfile=$(mktemp)
    for book in "${SHELF_BOOKS[@]}"; do
        echo "$book"
    done | sort -t: -k1,1 > "$tmpfile"
    cat "$tmpfile"
    rm -f "$tmpfile"
}

is_full() {
    [ "${#SHELF_BOOKS[@]}" -ge "$SHELF_CAPACITY" ]
}

generate_report() {
    local total=${#SHELF_BOOKS[@]}
    local available_count=0
    for book in "${SHELF_BOOKS[@]}"; do
        if [ "$(book_available "$book")" = "true" ]; then
            available_count=$((available_count + 1))
        fi
    done

    local avail_pct=0 cap_pct=0
    if [ "$total" -gt 0 ]; then
        avail_pct=$((available_count * 100 / total))
    fi
    if [ "$SHELF_CAPACITY" -gt 0 ]; then
        cap_pct=$((total * 100 / SHELF_CAPACITY))
    fi

    # Collect unique authors in order
    local -a author_order=()
    local -A author_counts=()
    for book in "${SHELF_BOOKS[@]}"; do
        local author
        author=$(book_author "$book")
        if [ -z "${author_counts[$author]+x}" ]; then
            author_order+=("$author")
            author_counts[$author]=0
        fi
        author_counts[$author]=$(( ${author_counts[$author]} + 1 ))
    done
    local unique_authors=${#author_order[@]}

    # Year range
    local min_year=0 max_year=0
    if [ "$total" -gt 0 ]; then
        min_year=$(book_year "${SHELF_BOOKS[0]}")
        max_year=$min_year
        for book in "${SHELF_BOOKS[@]}"; do
            local year
            year=$(book_year "$book")
            if [ "$year" -lt "$min_year" ]; then min_year=$year; fi
            if [ "$year" -gt "$max_year" ]; then max_year=$year; fi
        done
    fi

    echo "=== Library Report: ${SHELF_NAME} ==="
    echo "Total books: ${total}"
    echo "Available: ${available_count} / ${total} (${avail_pct}%)"
    echo "Capacity: ${total} / ${SHELF_CAPACITY} (${cap_pct}% full)"
    echo ""

    echo "Authors (${unique_authors} unique):"
    for author in "${author_order[@]}"; do
        echo "  - ${author} (${author_counts[$author]} books)"
    done
    echo ""

    echo "Year range: ${min_year} - ${max_year}"
    echo ""

    echo "Books by availability:"
    for book in "${SHELF_BOOKS[@]}"; do
        local marker
        if [ "$(book_available "$book")" = "true" ]; then
            marker="+"
        else
            marker="-"
        fi
        echo "  [${marker}] $(book_title "$book") by $(book_author "$book") ($(book_year "$book")) - $(book_isbn "$book")"
    done
}
