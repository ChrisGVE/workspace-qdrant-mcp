/**
 * @fileoverview Business logic for bookshelf storage operations.
 */

/**
 * Adds a book to a shelf.
 * @param {import('./models.js').Shelf} shelf - The target shelf
 * @param {import('./models.js').Book} book - The book to add
 * @throws {Error} If the shelf is at capacity
 */
export function addBook(shelf, book) {
  if (shelf.books.length >= shelf.capacity) {
    throw new Error(`Shelf "${shelf.name}" is at capacity (${shelf.capacity})`);
  }
  shelf.books.push(book);
}

/**
 * Removes a book from a shelf by ISBN.
 * @param {import('./models.js').Shelf} shelf - The target shelf
 * @param {string} isbn - ISBN of the book to remove
 * @throws {Error} If no book with that ISBN exists on the shelf
 */
export function removeBook(shelf, isbn) {
  const index = shelf.books.findIndex((b) => b.isbn === isbn);
  if (index === -1) {
    throw new Error(`Book with ISBN ${isbn} not found on shelf "${shelf.name}"`);
  }
  shelf.books.splice(index, 1);
}

/**
 * Finds books by author using case-insensitive substring matching.
 * @param {import('./models.js').Shelf} shelf - The shelf to search
 * @param {string} author - Author name or substring to search for
 * @returns {import('./models.js').Book[]} Books whose author matches
 */
export function findByAuthor(shelf, author) {
  const query = author.toLowerCase();
  return shelf.books.filter((b) => b.author.toLowerCase().includes(query));
}

/**
 * Finds books published within an inclusive year range.
 * @param {import('./models.js').Shelf} shelf - The shelf to search
 * @param {number} startYear - Start of the year range (inclusive)
 * @param {number} endYear - End of the year range (inclusive)
 * @returns {import('./models.js').Book[]} Books within the year range
 */
export function findByYearRange(shelf, startYear, endYear) {
  return shelf.books.filter((b) => b.year >= startYear && b.year <= endYear);
}

/**
 * Returns a sorted copy of all books on the shelf, ordered by title.
 * Does not mutate the original shelf.
 * @param {import('./models.js').Shelf} shelf - The shelf to sort
 * @returns {import('./models.js').Book[]} Sorted copy of the books array
 */
export function sortByTitle(shelf) {
  return [...shelf.books].sort((a, b) => a.title.localeCompare(b.title));
}

/**
 * Checks whether the shelf has reached its maximum capacity.
 * @param {import('./models.js').Shelf} shelf - The shelf to check
 * @returns {boolean} True if the shelf is full
 */
export function isFull(shelf) {
  return shelf.books.length >= shelf.capacity;
}

/**
 * Generates a multi-line formatted report for the shelf.
 * @param {import('./models.js').Shelf} shelf - The shelf to report on
 * @returns {string} Formatted library report
 */
export function generateReport(shelf) {
  const total = shelf.books.length;
  const available = shelf.books.filter((b) => b.available).length;
  const availablePct = total > 0 ? Math.round((available / total) * 100) : 0;
  const capacityPct = Math.round((total / shelf.capacity) * 100);

  const authorCounts = {};
  for (const book of shelf.books) {
    authorCounts[book.author] = (authorCounts[book.author] ?? 0) + 1;
  }
  const authorLines = Object.entries(authorCounts)
    .map(([name, count]) => `  - ${name} (${count} books)`)
    .join('\n');

  const years = shelf.books.map((b) => b.year);
  const minYear = Math.min(...years);
  const maxYear = Math.max(...years);

  const bookLines = shelf.books
    .map((b) => {
      const marker = b.available ? '[+]' : '[-]';
      return `  ${marker} ${b.title} by ${b.author} (${b.year}) - ${b.isbn}`;
    })
    .join('\n');

  return [
    `=== Library Report: ${shelf.name} ===`,
    `Total books: ${total}`,
    `Available: ${available} / ${total} (${availablePct}%)`,
    `Capacity: ${total} / ${shelf.capacity} (${capacityPct}% full)`,
    ``,
    `Authors (${Object.keys(authorCounts).length} unique):`,
    authorLines,
    ``,
    `Year range: ${minYear} - ${maxYear}`,
    ``,
    `Books by availability:`,
    bookLines,
  ].join('\n');
}
