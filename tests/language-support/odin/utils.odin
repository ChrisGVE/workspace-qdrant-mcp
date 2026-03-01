package main

import "core:fmt"
import "core:strconv"
import "core:strings"
import "core:unicode/utf8"

validate_isbn :: proc(isbn: string) -> bool {
    if len(isbn) != 13 do return false
    total := 0
    for i := 0; i < 13; i += 1 {
        d := int(isbn[i]) - int('0')
        if d < 0 || d > 9 do return false
        if i % 2 == 0 {
            total += d
        } else {
            total += d * 3
        }
    }
    return total % 10 == 0
}

format_book :: proc(b: Book) -> string {
    return fmt.aprintf(
        "\"%s\" by %s (%d) [ISBN: %s]",
        b.title, b.author, b.year, b.isbn,
    )
}

parse_csv_line :: proc(line: string) -> (Book, bool) {
    parts := strings.split(line, ",")
    defer delete(parts)
    if len(parts) != 5 do return Book{}, false

    year, ok := strconv.parse_int(strings.trim_space(parts[2]))
    if !ok do return Book{}, false

    avail_str := strings.to_lower(strings.trim_space(parts[4]))
    defer delete(avail_str)
    available: bool
    if avail_str == "true" {
        available = true
    } else if avail_str == "false" {
        available = false
    } else {
        return Book{}, false
    }

    return Book{
        title     = strings.trim_space(parts[0]),
        author    = strings.trim_space(parts[1]),
        year      = year,
        isbn      = strings.trim_space(parts[3]),
        available = available,
    }, true
}
