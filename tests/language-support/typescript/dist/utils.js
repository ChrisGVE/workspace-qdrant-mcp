import { createBook } from "./models.js";
/**
 * Validates an ISBN-13 using the standard check digit algorithm.
 * Steps:
 *   1. Strip non-digit characters; must be exactly 13 digits.
 *   2. Alternate multiply digits by 1 and 3, summing all products.
 *   3. Valid when sum mod 10 === 0.
 */
export function validateIsbn(isbn) {
    const digits = isbn.replace(/[^0-9]/g, "");
    if (digits.length !== 13) {
        return false;
    }
    let sum = 0;
    for (let i = 0; i < 13; i++) {
        const digit = parseInt(digits[i], 10);
        const multiplier = i % 2 === 0 ? 1 : 3;
        sum += digit * multiplier;
    }
    return sum % 10 === 0;
}
/**
 * Formats a book as a single descriptive line.
 * Example: `"Title" by Author (YYYY) [ISBN: XXXXXXXXXXXXX]`
 */
export function formatBook(book) {
    return `"${book.title}" by ${book.author} (${book.year}) [ISBN: ${book.isbn}]`;
}
/**
 * Parses a CSV line with fields: title,author,year,isbn,available.
 * Throws an Error for wrong field count, non-numeric year, or invalid boolean.
 */
export function parseCsvLine(line) {
    const fields = line.split(",");
    if (fields.length !== 5) {
        throw new Error(`Expected 5 fields, got ${fields.length}: "${line}"`);
    }
    const [title, author, yearStr, isbn, availableStr] = fields;
    const year = parseInt(yearStr, 10);
    if (isNaN(year)) {
        throw new Error(`Invalid year value: "${yearStr}"`);
    }
    if (availableStr !== "true" && availableStr !== "false") {
        throw new Error(`Invalid boolean value for available: "${availableStr}"`);
    }
    const available = availableStr === "true";
    return createBook(title, author, year, isbn, available);
}
//# sourceMappingURL=utils.js.map