"""Storage operations for managing books on shelves."""

from collections import Counter

from .models import Book, Shelf


def add_book(shelf: Shelf, book: Book) -> None:
    """Add a book to the shelf.

    Args:
        shelf: The shelf to add to.
        book: The book to add.

    Raises:
        ValueError: If the shelf is at capacity.
    """
    if is_full(shelf):
        raise ValueError(
            f"Shelf '{shelf.name}' is at capacity ({shelf.capacity})"
        )
    shelf.books.append(book)


def remove_book(shelf: Shelf, isbn: str) -> Book:
    """Remove a book from the shelf by ISBN.

    Args:
        shelf: The shelf to remove from.
        isbn: The ISBN of the book to remove.

    Returns:
        The removed book.

    Raises:
        ValueError: If no book with the given ISBN is found.
    """
    for i, book in enumerate(shelf.books):
        if book.isbn == isbn:
            return shelf.books.pop(i)
    raise ValueError(f"No book with ISBN '{isbn}' found on shelf")


def find_by_author(shelf: Shelf, author: str) -> list[Book]:
    """Find books by author using case-insensitive substring match.

    Args:
        shelf: The shelf to search.
        author: The author name or substring to match.

    Returns:
        List of matching books.
    """
    query = author.lower()
    return [b for b in shelf.books if query in b.author.lower()]


def find_by_year_range(
    shelf: Shelf, start_year: int, end_year: int
) -> list[Book]:
    """Find books published within an inclusive year range.

    Args:
        shelf: The shelf to search.
        start_year: The start of the year range (inclusive).
        end_year: The end of the year range (inclusive).

    Returns:
        List of matching books.
    """
    return [b for b in shelf.books if start_year <= b.year <= end_year]


def sort_by_title(shelf: Shelf) -> list[Book]:
    """Return books sorted by title without mutating the shelf.

    Args:
        shelf: The shelf whose books to sort.

    Returns:
        A new list of books sorted alphabetically by title.
    """
    return sorted(shelf.books, key=lambda b: b.title)


def is_full(shelf: Shelf) -> bool:
    """Check whether the shelf has reached its capacity.

    Args:
        shelf: The shelf to check.

    Returns:
        True if the shelf is at or over capacity.
    """
    return len(shelf.books) >= shelf.capacity


def generate_report(shelf: Shelf) -> str:
    """Generate a multi-line formatted report for the shelf.

    Args:
        shelf: The shelf to report on.

    Returns:
        A formatted report string.
    """
    total = len(shelf.books)
    available_count = sum(1 for b in shelf.books if b.available)
    avail_pct = (available_count * 100 // total) if total > 0 else 0
    cap_pct = (total * 100 // shelf.capacity) if shelf.capacity > 0 else 0

    author_counts: Counter[str] = Counter(b.author for b in shelf.books)
    unique_authors = len(author_counts)

    years = [b.year for b in shelf.books]
    min_year = min(years) if years else 0
    max_year = max(years) if years else 0

    lines: list[str] = []
    lines.append(f"=== Library Report: {shelf.name} ===")
    lines.append(f"Total books: {total}")
    lines.append(f"Available: {available_count} / {total} ({avail_pct}%)")
    lines.append(f"Capacity: {total} / {shelf.capacity} ({cap_pct}% full)")
    lines.append("")

    lines.append(f"Authors ({unique_authors} unique):")
    for author, count in author_counts.items():
        lines.append(f"  - {author} ({count} books)")
    lines.append("")

    lines.append(f"Year range: {min_year} - {max_year}")
    lines.append("")

    lines.append("Books by availability:")
    for book in shelf.books:
        marker = "+" if book.available else "-"
        lines.append(
            f"  [{marker}] {book.title} by {book.author} "
            f"({book.year}) - {book.isbn}"
        )

    return "\n".join(lines)
