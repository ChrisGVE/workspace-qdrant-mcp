#!/usr/bin/env bash
# Utils: ISBN validation, book formatting, CSV parsing

validate_isbn() {
    local isbn="$1"
    if [ ${#isbn} -ne 13 ]; then
        echo "false"
        return
    fi
    if ! [[ "$isbn" =~ ^[0-9]{13}$ ]]; then
        echo "false"
        return
    fi
    local total=0
    for ((i = 0; i < 13; i++)); do
        local digit=${isbn:$i:1}
        if (( i % 2 == 0 )); then
            total=$((total + digit))
        else
            total=$((total + digit * 3))
        fi
    done
    if (( total % 10 == 0 )); then
        echo "true"
    else
        echo "false"
    fi
}

format_book() {
    local book="$1"
    local title author year isbn
    title=$(book_title "$book")
    author=$(book_author "$book")
    year=$(book_year "$book")
    isbn=$(book_isbn "$book")
    echo "\"${title}\" by ${author} (${year}) [ISBN: ${isbn}]"
}

parse_csv_line() {
    local line="$1"
    IFS=',' read -r title author year_str isbn avail_str <<< "$line"

    # Trim whitespace
    title=$(echo "$title" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    author=$(echo "$author" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    year_str=$(echo "$year_str" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    isbn=$(echo "$isbn" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    avail_str=$(echo "$avail_str" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')

    if ! [[ "$year_str" =~ ^[0-9]+$ ]]; then
        echo "Error: Invalid year: ${year_str}" >&2
        return 1
    fi

    local avail_lower
    avail_lower=$(echo "$avail_str" | tr '[:upper:]' '[:lower:]')
    if [ "$avail_lower" != "true" ] && [ "$avail_lower" != "false" ]; then
        echo "Error: Invalid boolean: ${avail_str}" >&2
        return 1
    fi

    make_book "$title" "$author" "$year_str" "$isbn" "$avail_lower"
}
