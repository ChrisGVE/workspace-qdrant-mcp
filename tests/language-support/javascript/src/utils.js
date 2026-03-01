/**
 * @fileoverview Utility functions for ISBN validation, book formatting, and CSV parsing.
 */

import { Book } from './models.js';

/**
 * Validates an ISBN-13 number using the standard check digit algorithm.
 * The algorithm alternates multiplying digits by 1 and 3, sums the products,
 * and checks that the result is divisible by 10.
 * @param {string} isbn - The ISBN string to validate (must be exactly 13 digits)
 * @returns {boolean} True if the ISBN-13 check digit is valid
 */
export function validateIsbn(isbn) {
  if (!/^\d{13}$/.test(isbn)) {
    return false;
  }
  let sum = 0;
  for (let i = 0; i < 13; i++) {
    const digit = parseInt(isbn[i], 10);
    sum += i % 2 === 0 ? digit : digit * 3;
  }
  return sum % 10 === 0;
}

/**
 * Formats a book as a single descriptive line.
 * @param {Book} book - The book to format
 * @returns {string} Formatted string: "Title" by Author (YYYY) [ISBN: XXXXXXXXXXXXX]
 */
export function formatBook(book) {
  return `"${book.title}" by ${book.author} (${book.year}) [ISBN: ${book.isbn}]`;
}

/**
 * Parses a CSV line into a Book object.
 * Expected format: title,author,year,isbn,available
 * @param {string} line - CSV line to parse
 * @returns {Book} The parsed Book instance
 * @throws {Error} If the line has wrong field count, invalid year, or invalid boolean
 */
export function parseCsvLine(line) {
  const fields = line.split(',');
  if (fields.length !== 5) {
    throw new Error(`Expected 5 fields, got ${fields.length}`);
  }
  const [title, author, yearStr, isbn, availableStr] = fields;
  const year = parseInt(yearStr.trim(), 10);
  if (isNaN(year)) {
    throw new Error(`Invalid year: "${yearStr.trim()}"`);
  }
  const availableLower = availableStr.trim().toLowerCase();
  if (availableLower !== 'true' && availableLower !== 'false') {
    throw new Error(`Invalid boolean for available: "${availableStr.trim()}"`);
  }
  return new Book(title.trim(), author.trim(), year, isbn.trim(), availableLower === 'true');
}
