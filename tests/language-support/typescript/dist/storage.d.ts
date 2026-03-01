import type { Book, Shelf } from "./models.js";
/**
 * Adds a book to the shelf.
 * Throws if the shelf is already at capacity.
 */
export declare function addBook(shelf: Shelf, book: Book): void;
/**
 * Removes a book from the shelf by ISBN.
 * Throws if no book with the given ISBN is found.
 */
export declare function removeBook(shelf: Shelf, isbn: string): void;
/**
 * Returns all books whose author contains the query string (case-insensitive).
 */
export declare function findByAuthor(shelf: Shelf, author: string): Book[];
/**
 * Returns all books published within the inclusive year range [startYear, endYear].
 */
export declare function findByYearRange(shelf: Shelf, startYear: number, endYear: number): Book[];
/**
 * Returns a copy of the shelf's books sorted alphabetically by title.
 * The original shelf is not mutated.
 */
export declare function sortByTitle(shelf: Shelf): Book[];
/** Returns true when the shelf has reached its maximum capacity. */
export declare function isFull(shelf: Shelf): boolean;
/**
 * Generates a formatted multi-line report summarising the shelf contents.
 * Includes totals, availability, unique authors, year range, and per-book listing.
 */
export declare function generateReport(shelf: Shelf): string;
