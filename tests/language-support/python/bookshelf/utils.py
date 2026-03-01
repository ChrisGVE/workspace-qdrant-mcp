"""Utility functions for ISBN validation, formatting, and CSV parsing."""

from .models import Book


def validate_isbn(isbn: str) -> bool:
    """Validate an ISBN-13 check digit.

    Alternately multiplies digits by 1 and 3, sums all products,
    and checks whether the total is divisible by 10.

    Args:
        isbn: The ISBN string to validate.

    Returns:
        True if the ISBN-13 check digit is valid.
    """
    if len(isbn) != 13 or not isbn.isdigit():
        return False
    total = sum(
        int(ch) * (1 if i % 2 == 0 else 3) for i, ch in enumerate(isbn)
    )
    return total % 10 == 0


def format_book(book: Book) -> str:
    """Format a book as a single descriptive line.

    Args:
        book: The book to format.

    Returns:
        Formatted string: "Title" by Author (YYYY) [ISBN: XXXXXXXXXXXXX]
    """
    return (
        f'"{book.title}" by {book.author} '
        f"({book.year}) [ISBN: {book.isbn}]"
    )


def parse_csv_line(line: str) -> Book:
    """Parse a CSV line into a Book.

    Expected format: title,author,year,isbn,available

    Args:
        line: The CSV line to parse.

    Returns:
        A Book instance parsed from the line.

    Raises:
        ValueError: On wrong field count, invalid year, or invalid boolean.
    """
    parts = line.split(",")
    if len(parts) != 5:
        raise ValueError(f"Expected 5 fields, got {len(parts)}")
    title, author, year_str, isbn, avail_str = parts

    try:
        year = int(year_str)
    except ValueError:
        raise ValueError(f"Invalid year: {year_str}")

    avail_lower = avail_str.strip().lower()
    if avail_lower == "true":
        available = True
    elif avail_lower == "false":
        available = False
    else:
        raise ValueError(f"Invalid boolean: {avail_str}")

    return Book(
        title=title.strip(),
        author=author.strip(),
        year=year,
        isbn=isbn.strip(),
        available=available,
    )
