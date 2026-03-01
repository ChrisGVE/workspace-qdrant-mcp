/** Genre classification for books in the library. */
export var Genre;
(function (Genre) {
    Genre["Fiction"] = "Fiction";
    Genre["NonFiction"] = "NonFiction";
    Genre["Science"] = "Science";
    Genre["History"] = "History";
    Genre["Biography"] = "Biography";
    Genre["Technology"] = "Technology";
    Genre["Philosophy"] = "Philosophy";
})(Genre || (Genre = {}));
/**
 * Factory function for creating a Book.
 * All fields are required; available indicates whether the book can be borrowed.
 */
export function createBook(title, author, year, isbn, available) {
    return { title, author, year, isbn, available };
}
/**
 * Factory function for creating an empty Shelf.
 * Capacity defines the maximum number of books the shelf can hold.
 */
export function createShelf(name, capacity) {
    return { name, capacity, books: [] };
}
//# sourceMappingURL=models.js.map