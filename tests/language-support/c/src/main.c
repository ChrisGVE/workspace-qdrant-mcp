#include <stdio.h>
#include "models.h"
#include "storage.h"
#include "utils.h"

int main(void) {
    Shelf shelf = create_shelf("Computer Science", 10);

    Book books[] = {
        create_book("The Art of Computer Programming", "Donald Knuth",
                    1968, "9780201896831", 1),
        create_book("Structure and Interpretation of Computer Programs",
                    "Harold Abelson", 1996, "9780262510875", 1),
        create_book("Introduction to Algorithms", "Thomas Cormen",
                    2009, "9780262033848", 0),
        create_book("Design Patterns", "Erich Gamma",
                    1994, "9780201633610", 1),
        create_book("The Pragmatic Programmer", "David Thomas",
                    2019, "9780135957059", 1),
    };
    int num_books = sizeof(books) / sizeof(books[0]);

    for (int i = 0; i < num_books; i++) {
        add_book(&shelf, books[i]);
    }

    char report[4096];
    generate_report(&shelf, report, sizeof(report));
    printf("%s\n", report);

    printf("--- Search by author \"knuth\" ---\n");
    Book results[10];
    int found = find_by_author(&shelf, "knuth", results, 10);
    for (int i = 0; i < found; i++) {
        char fmt[512];
        format_book(&results[i], fmt, sizeof(fmt));
        printf("  %s\n", fmt);
    }
    printf("\n");

    printf("--- Search by year range 1990-2010 ---\n");
    found = find_by_year_range(&shelf, 1990, 2010, results, 10);
    for (int i = 0; i < found; i++) {
        char fmt[512];
        format_book(&results[i], fmt, sizeof(fmt));
        printf("  %s\n", fmt);
    }
    printf("\n");

    printf("--- Parse CSV ---\n");
    Book parsed;
    if (parse_csv_line("Clean Code,Robert Martin,2008,9780132350884,true",
                       &parsed) == 0) {
        char fmt[512];
        format_book(&parsed, fmt, sizeof(fmt));
        printf("  Parsed: %s\n", fmt);
    }
    printf("\n");

    printf("--- ISBN Validation ---\n");
    for (int i = 0; i < num_books; i++) {
        printf("  %s: %s\n", books[i].isbn,
               validate_isbn(books[i].isbn) ? "valid" : "invalid");
    }

    free_shelf(&shelf);
    return 0;
}
