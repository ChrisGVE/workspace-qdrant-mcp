/**
 * @fileoverview Data models for the bookshelf library management system.
 */

/**
 * Represents a book in the library.
 */
export class Book {
  /**
   * @param {string} title - The book title
   * @param {string} author - The author name
   * @param {number} year - Publication year
   * @param {string} isbn - ISBN-13 identifier
   * @param {boolean} available - Whether the book is available for borrowing
   */
  constructor(title, author, year, isbn, available) {
    this.title = title;
    this.author = author;
    this.year = year;
    this.isbn = isbn;
    this.available = available;
  }
}

/**
 * Represents a shelf that holds books.
 */
export class Shelf {
  /**
   * @param {string} name - The shelf name
   * @param {number} capacity - Maximum number of books the shelf can hold
   */
  constructor(name, capacity) {
    this.name = name;
    this.capacity = capacity;
    this.books = [];
  }
}

/**
 * Enumeration of book genres.
 * @readonly
 * @enum {string}
 */
export const Genre = Object.freeze({
  Fiction: 'Fiction',
  NonFiction: 'NonFiction',
  Science: 'Science',
  History: 'History',
  Biography: 'Biography',
  Technology: 'Technology',
  Philosophy: 'Philosophy',
});
