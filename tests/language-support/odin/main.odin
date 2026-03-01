package main

import "core:fmt"

main :: proc() {
    s := create_shelf("Computer Science", 10)

    books := [?]Book{
        create_book("The Art of Computer Programming",
                    "Donald Knuth", 1968, "9780201896831", true),
        create_book("Structure and Interpretation of Computer Programs",
                    "Harold Abelson", 1996, "9780262510875", true),
        create_book("Introduction to Algorithms",
                    "Thomas Cormen", 2009, "9780262033848", false),
        create_book("Design Patterns",
                    "Erich Gamma", 1994, "9780201633610", true),
        create_book("The Pragmatic Programmer",
                    "David Thomas", 2019, "9780135957059", true),
    }

    for &b in books {
        add_book(&s, b)
    }

    report := generate_report(s)
    fmt.printf("%s", report)
    fmt.println()

    fmt.println("--- Search by author \"knuth\" ---")
    author_results := find_by_author(s, "knuth")
    defer delete(author_results)
    for b in author_results {
        formatted := format_book(b)
        defer delete(formatted)
        fmt.printf("  %s\n", formatted)
    }
    fmt.println()

    fmt.println("--- Search by year range 1990-2010 ---")
    year_results := find_by_year_range(s, 1990, 2010)
    defer delete(year_results)
    for b in year_results {
        formatted := format_book(b)
        defer delete(formatted)
        fmt.printf("  %s\n", formatted)
    }
    fmt.println()

    fmt.println("--- Parse CSV ---")
    parsed, ok := parse_csv_line("Clean Code,Robert Martin,2008,9780132350884,true")
    if ok {
        formatted := format_book(parsed)
        defer delete(formatted)
        fmt.printf("  Parsed: %s\n", formatted)
    }
    fmt.println()

    fmt.println("--- ISBN Validation ---")
    for b in books {
        status := "valid" if validate_isbn(b.isbn) else "invalid"
        fmt.printf("  %s: %s\n", b.isbn, status)
    }
}
