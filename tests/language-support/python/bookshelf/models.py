"""Data models for the bookshelf library management system.

Provides Book and Shelf dataclasses and a Genre enumeration
for classifying books in the library.
"""

from dataclasses import dataclass, field
from enum import Enum


class Genre(Enum):
    """Genre classification for books.

    Each variant represents a distinct literary category
    used for organizing books on shelves.
    """

    FICTION = "Fiction"
    NON_FICTION = "NonFiction"
    SCIENCE = "Science"
    HISTORY = "History"
    BIOGRAPHY = "Biography"
    TECHNOLOGY = "Technology"
    PHILOSOPHY = "Philosophy"


@dataclass
class Book:
    """Represents a single book in the library.

    Attributes:
        title: The title of the book.
        author: The author of the book.
        year: The year the book was published.
        isbn: The ISBN-13 identifier.
        available: Whether the book is currently available.
    """

    title: str
    author: str
    year: int
    isbn: str
    available: bool


@dataclass
class Shelf:
    """Represents a bookshelf with a fixed capacity.

    Attributes:
        name: The name of the shelf.
        capacity: The maximum number of books the shelf can hold.
        books: The list of books currently on the shelf.
    """

    name: str
    capacity: int
    books: list[Book] = field(default_factory=list)
