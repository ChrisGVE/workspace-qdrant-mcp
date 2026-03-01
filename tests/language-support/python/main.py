"""Bookshelf demonstration: creates a shelf, adds books, and runs queries."""

from bookshelf.models import Book, Shelf
from bookshelf.storage import (
    add_book,
    find_by_author,
    find_by_year_range,
    generate_report,
)
from bookshelf.utils import format_book, parse_csv_line, validate_isbn


def main() -> None:
    """Run the bookshelf demonstration."""
    shelf = Shelf(name="Computer Science", capacity=10)

    books = [
        Book("The Art of Computer Programming", "Donald Knuth", 1968, "9780201896831", True),
        Book("Structure and Interpretation of Computer Programs", "Harold Abelson", 1996, "9780262510875", True),
        Book("Introduction to Algorithms", "Thomas Cormen", 2009, "9780262033848", False),
        Book("Design Patterns", "Erich Gamma", 1994, "9780201633610", True),
        Book("The Pragmatic Programmer", "David Thomas", 2019, "9780135957059", True),
    ]
    for book in books:
        add_book(shelf, book)

    print(generate_report(shelf))
    print()

    print('--- Search by author "knuth" ---')
    for book in find_by_author(shelf, "knuth"):
        print(f"  {format_book(book)}")
    print()

    print("--- Search by year range 1990-2010 ---")
    for book in find_by_year_range(shelf, 1990, 2010):
        print(f"  {format_book(book)}")
    print()

    print("--- Parse CSV ---")
    parsed = parse_csv_line("Clean Code,Robert Martin,2008,9780132350884,true")
    print(f"  Parsed: {format_book(parsed)}")
    print()

    print("--- ISBN Validation ---")
    for book in books:
        status = "valid" if validate_isbn(book.isbn) else "invalid"
        print(f"  {book.isbn}: {status}")


if __name__ == "__main__":
    main()
