#include <iostream>
#include "models.hpp"
#include "storage.hpp"
#include "utils.hpp"

int main() {
    Shelf shelf("Computer Science", 10);

    shelf.books.clear();
    add_book(shelf, Book("The Art of Computer Programming",
                         "Donald Knuth", 1968, "9780201896831", true));
    add_book(shelf, Book("Structure and Interpretation of Computer Programs",
                         "Harold Abelson", 1996, "9780262510875", true));
    add_book(shelf, Book("Introduction to Algorithms",
                         "Thomas Cormen", 2009, "9780262033848", false));
    add_book(shelf, Book("Design Patterns",
                         "Erich Gamma", 1994, "9780201633610", true));
    add_book(shelf, Book("The Pragmatic Programmer",
                         "David Thomas", 2019, "9780135957059", true));

    std::string report = generate_report(shelf);
    /* Remove trailing newline from report since we print one */
    if (!report.empty() && report.back() == '\n') {
        report.pop_back();
    }
    std::cout << report << std::endl;

    std::cout << std::endl;
    std::cout << "--- Search by author \"knuth\" ---" << std::endl;
    for (const auto &book : find_by_author(shelf, "knuth")) {
        std::cout << "  " << format_book(book) << std::endl;
    }
    std::cout << std::endl;

    std::cout << "--- Search by year range 1990-2010 ---" << std::endl;
    for (const auto &book : find_by_year_range(shelf, 1990, 2010)) {
        std::cout << "  " << format_book(book) << std::endl;
    }
    std::cout << std::endl;

    std::cout << "--- Parse CSV ---" << std::endl;
    Book parsed = parse_csv_line(
        "Clean Code,Robert Martin,2008,9780132350884,true");
    std::cout << "  Parsed: " << format_book(parsed) << std::endl;
    std::cout << std::endl;

    std::cout << "--- ISBN Validation ---" << std::endl;
    std::vector<std::string> isbns = {
        "9780201896831", "9780262510875", "9780262033848",
        "9780201633610", "9780135957059"
    };
    for (const auto &isbn : isbns) {
        std::cout << "  " << isbn << ": "
                  << (validate_isbn(isbn) ? "valid" : "invalid")
                  << std::endl;
    }

    return 0;
}
