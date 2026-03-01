/**
 * Adds a book to the shelf.
 * Throws if the shelf is already at capacity.
 */
export function addBook(shelf, book) {
    if (shelf.books.length >= shelf.capacity) {
        throw new Error(`Shelf "${shelf.name}" is at capacity (${shelf.capacity} books)`);
    }
    shelf.books.push(book);
}
/**
 * Removes a book from the shelf by ISBN.
 * Throws if no book with the given ISBN is found.
 */
export function removeBook(shelf, isbn) {
    const index = shelf.books.findIndex((b) => b.isbn === isbn);
    if (index === -1) {
        throw new Error(`Book with ISBN ${isbn} not found on shelf "${shelf.name}"`);
    }
    shelf.books.splice(index, 1);
}
/**
 * Returns all books whose author contains the query string (case-insensitive).
 */
export function findByAuthor(shelf, author) {
    const query = author.toLowerCase();
    return shelf.books.filter((b) => b.author.toLowerCase().includes(query));
}
/**
 * Returns all books published within the inclusive year range [startYear, endYear].
 */
export function findByYearRange(shelf, startYear, endYear) {
    return shelf.books.filter((b) => b.year >= startYear && b.year <= endYear);
}
/**
 * Returns a copy of the shelf's books sorted alphabetically by title.
 * The original shelf is not mutated.
 */
export function sortByTitle(shelf) {
    return [...shelf.books].sort((a, b) => a.title.localeCompare(b.title));
}
/** Returns true when the shelf has reached its maximum capacity. */
export function isFull(shelf) {
    return shelf.books.length >= shelf.capacity;
}
/**
 * Generates a formatted multi-line report summarising the shelf contents.
 * Includes totals, availability, unique authors, year range, and per-book listing.
 */
export function generateReport(shelf) {
    const lines = [];
    const total = shelf.books.length;
    const available = shelf.books.filter((b) => b.available).length;
    const availablePct = total > 0 ? Math.round((available / total) * 100) : 0;
    const capacityPct = shelf.capacity > 0
        ? Math.round((total / shelf.capacity) * 100)
        : 0;
    lines.push(`=== Library Report: ${shelf.name} ===`);
    lines.push(`Total books: ${total}`);
    lines.push(`Available: ${available} / ${total} (${availablePct}%)`);
    lines.push(`Capacity: ${total} / ${shelf.capacity} (${capacityPct}% full)`);
    lines.push("");
    const authorCounts = new Map();
    for (const book of shelf.books) {
        authorCounts.set(book.author, (authorCounts.get(book.author) ?? 0) + 1);
    }
    lines.push(`Authors (${authorCounts.size} unique):`);
    for (const [author, count] of authorCounts) {
        lines.push(`  - ${author} (${count} books)`);
    }
    lines.push("");
    if (total > 0) {
        const years = shelf.books.map((b) => b.year);
        lines.push(`Year range: ${Math.min(...years)} - ${Math.max(...years)}`);
    }
    else {
        lines.push("Year range: N/A");
    }
    lines.push("");
    lines.push("Books by availability:");
    for (const book of shelf.books) {
        const marker = book.available ? "[+]" : "[-]";
        lines.push(`  ${marker} ${book.title} by ${book.author} (${book.year}) - ${book.isbn}`);
    }
    lines.push("");
    return lines.join("\n");
}
//# sourceMappingURL=storage.js.map