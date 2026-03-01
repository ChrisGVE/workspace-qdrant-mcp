package main

import "core:fmt"
import "core:strings"
import "core:slice"

add_book :: proc(s: ^Shelf, b: Book) -> bool {
    if is_full(s^) do return false
    append(&s.books, b)
    return true
}

remove_book :: proc(s: ^Shelf, isbn: string) -> bool {
    for i := 0; i < len(s.books); i += 1 {
        if s.books[i].isbn == isbn {
            ordered_remove(&s.books, i)
            return true
        }
    }
    return false
}

find_by_author :: proc(s: Shelf, author: string) -> [dynamic]Book {
    result := make([dynamic]Book, 0)
    query := strings.to_lower(author)
    defer delete(query)
    for b in s.books {
        lower_author := strings.to_lower(b.author)
        defer delete(lower_author)
        if strings.contains(lower_author, query) {
            append(&result, b)
        }
    }
    return result
}

find_by_year_range :: proc(s: Shelf, start_year, end_year: int) -> [dynamic]Book {
    result := make([dynamic]Book, 0)
    for b in s.books {
        if b.year >= start_year && b.year <= end_year {
            append(&result, b)
        }
    }
    return result
}

sort_by_title :: proc(s: Shelf) -> [dynamic]Book {
    result := make([dynamic]Book, len(s.books))
    for i := 0; i < len(s.books); i += 1 {
        result[i] = s.books[i]
    }
    slice.sort_by(result[:], proc(a, b: Book) -> bool {
        return a.title < b.title
    })
    return result
}

is_full :: proc(s: Shelf) -> bool {
    return len(s.books) >= s.capacity
}

generate_report :: proc(s: Shelf) -> string {
    total := len(s.books)
    avail_count := 0
    min_year := 9999
    max_year := 0

    Author_Entry :: struct { name: string, count: int }
    authors := make([dynamic]Author_Entry, 0)
    defer delete(authors)

    for b in s.books {
        if b.available do avail_count += 1
        if b.year < min_year do min_year = b.year
        if b.year > max_year do max_year = b.year
        found := false
        for &a in authors {
            if a.name == b.author {
                a.count += 1
                found = true
                break
            }
        }
        if !found {
            append(&authors, Author_Entry{b.author, 1})
        }
    }

    avail_pct := 0
    if total > 0 do avail_pct = avail_count * 100 / total
    cap_pct := 0
    if s.capacity > 0 do cap_pct = total * 100 / s.capacity

    b := strings.builder_make()

    fmt.sbprintf(&b, "=== Library Report: %s ===\n", s.name)
    fmt.sbprintf(&b, "Total books: %d\n", total)
    fmt.sbprintf(&b, "Available: %d / %d (%d%%)\n",
                 avail_count, total, avail_pct)
    fmt.sbprintf(&b, "Capacity: %d / %d (%d%% full)\n",
                 total, s.capacity, cap_pct)
    fmt.sbprintf(&b, "\n")
    fmt.sbprintf(&b, "Authors (%d unique):\n", len(authors))
    for a in authors {
        fmt.sbprintf(&b, "  - %s (%d books)\n", a.name, a.count)
    }
    fmt.sbprintf(&b, "\n")
    fmt.sbprintf(&b, "Year range: %d - %d\n", min_year, max_year)
    fmt.sbprintf(&b, "\n")
    fmt.sbprintf(&b, "Books by availability:\n")
    for book in s.books {
        marker := "+" if book.available else "-"
        fmt.sbprintf(&b, "  [%s] %s by %s (%d) - %s\n",
                     marker, book.title, book.author, book.year, book.isbn)
    }

    return strings.to_string(b)
}
