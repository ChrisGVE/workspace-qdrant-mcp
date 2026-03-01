import type { Book } from "./models.js";
/**
 * Validates an ISBN-13 using the standard check digit algorithm.
 * Steps:
 *   1. Strip non-digit characters; must be exactly 13 digits.
 *   2. Alternate multiply digits by 1 and 3, summing all products.
 *   3. Valid when sum mod 10 === 0.
 */
export declare function validateIsbn(isbn: string): boolean;
/**
 * Formats a book as a single descriptive line.
 * Example: `"Title" by Author (YYYY) [ISBN: XXXXXXXXXXXXX]`
 */
export declare function formatBook(book: Book): string;
/**
 * Parses a CSV line with fields: title,author,year,isbn,available.
 * Throws an Error for wrong field count, non-numeric year, or invalid boolean.
 */
export declare function parseCsvLine(line: string): Book;
