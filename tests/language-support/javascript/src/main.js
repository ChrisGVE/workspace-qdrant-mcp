/**
 * @fileoverview Entry point demonstrating the bookshelf library management system.
 */

import { Book, Shelf, Genre } from './models.js';
import { addBook, findByAuthor, findByYearRange, generateReport } from './storage.js';
import { validateIsbn, formatBook, parseCsvLine } from './utils.js';

// Create the shelf
const shelf = new Shelf('Computer Science', 10);

// Add the five canonical books
addBook(shelf, new Book('The Art of Computer Programming', 'Donald Knuth', 1968, '9780201896831', true));
addBook(shelf, new Book('Structure and Interpretation of Computer Programs', 'Harold Abelson', 1996, '9780262510875', true));
addBook(shelf, new Book('Introduction to Algorithms', 'Thomas Cormen', 2009, '9780262033848', false));
addBook(shelf, new Book('Design Patterns', 'Erich Gamma', 1994, '9780201633610', true));
addBook(shelf, new Book('The Pragmatic Programmer', 'David Thomas', 2019, '9780135957059', true));

// Print the full library report
console.log(generateReport(shelf));

// Search by author (case-insensitive)
console.log('\n--- Search by author "knuth" ---');
const byKnuth = findByAuthor(shelf, 'knuth');
for (const book of byKnuth) {
  console.log(' ', formatBook(book));
}

// Search by year range
console.log('\n--- Search by year range 1990-2010 ---');
const inRange = findByYearRange(shelf, 1990, 2010);
for (const book of inRange) {
  console.log(' ', formatBook(book));
}

// Parse a CSV line
console.log('\n--- Parse CSV ---');
const parsed = parseCsvLine('Clean Code,Robert Martin,2008,9780132350884,true');
console.log('  Parsed:', formatBook(parsed));

// Validate ISBNs for all books on the shelf
console.log('\n--- ISBN Validation ---');
for (const book of shelf.books) {
  const status = validateIsbn(book.isbn) ? 'valid' : 'invalid';
  console.log(`  ${book.isbn}: ${status}`);
}

// Suppress unused import warning — Genre is exported from models for completeness
void Genre;
