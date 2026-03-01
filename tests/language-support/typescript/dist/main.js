import { createBook, createShelf } from "./models.js";
import { addBook, findByAuthor, findByYearRange, generateReport, } from "./storage.js";
import { formatBook, parseCsvLine, validateIsbn } from "./utils.js";
const shelf = createShelf("Computer Science", 10);
addBook(shelf, createBook("The Art of Computer Programming", "Donald Knuth", 1968, "9780201896831", true));
addBook(shelf, createBook("Structure and Interpretation of Computer Programs", "Harold Abelson", 1996, "9780262510875", true));
addBook(shelf, createBook("Introduction to Algorithms", "Thomas Cormen", 2009, "9780262033848", false));
addBook(shelf, createBook("Design Patterns", "Erich Gamma", 1994, "9780201633610", true));
addBook(shelf, createBook("The Pragmatic Programmer", "David Thomas", 2019, "9780135957059", true));
console.log(generateReport(shelf));
console.log('--- Search by author "knuth" ---');
const byKnuth = findByAuthor(shelf, "knuth");
for (const book of byKnuth) {
    console.log(`  ${formatBook(book)}`);
}
console.log("");
console.log("--- Search by year range 1990-2010 ---");
const byYear = findByYearRange(shelf, 1990, 2010);
for (const book of byYear) {
    console.log(`  ${formatBook(book)}`);
}
console.log("");
console.log("--- Parse CSV ---");
const parsed = parseCsvLine("Clean Code,Robert Martin,2008,9780132350884,true");
console.log(`  Parsed: ${formatBook(parsed)}`);
console.log("");
console.log("--- ISBN Validation ---");
for (const book of shelf.books) {
    const result = validateIsbn(book.isbn) ? "valid" : "invalid";
    console.log(`  ${book.isbn}: ${result}`);
}
//# sourceMappingURL=main.js.map