/** Genre classification for books in the library. */
export enum Genre {
  Fiction = "Fiction",
  NonFiction = "NonFiction",
  Science = "Science",
  History = "History",
  Biography = "Biography",
  Technology = "Technology",
  Philosophy = "Philosophy",
}

/** Represents a single book in the library. */
export interface Book {
  title: string;
  author: string;
  year: number;
  isbn: string;
  available: boolean;
}

/** A named shelf with a fixed capacity holding a collection of books. */
export interface Shelf {
  name: string;
  capacity: number;
  books: Book[];
}

/**
 * Factory function for creating a Book.
 * All fields are required; available indicates whether the book can be borrowed.
 */
export function createBook(
  title: string,
  author: string,
  year: number,
  isbn: string,
  available: boolean
): Book {
  return { title, author, year, isbn, available };
}

/**
 * Factory function for creating an empty Shelf.
 * Capacity defines the maximum number of books the shelf can hold.
 */
export function createShelf(name: string, capacity: number): Shelf {
  return { name, capacity, books: [] };
}
