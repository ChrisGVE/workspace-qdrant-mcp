#include "models.h"
#include <stdlib.h>
#include <string.h>

Book create_book(const char *title, const char *author, int year,
                 const char *isbn, bool available) {
    Book book;
    strncpy(book.title, title, sizeof(book.title) - 1);
    book.title[sizeof(book.title) - 1] = '\0';
    strncpy(book.author, author, sizeof(book.author) - 1);
    book.author[sizeof(book.author) - 1] = '\0';
    book.year = year;
    strncpy(book.isbn, isbn, sizeof(book.isbn) - 1);
    book.isbn[sizeof(book.isbn) - 1] = '\0';
    book.available = available;
    return book;
}

Shelf create_shelf(const char *name, int capacity) {
    Shelf shelf;
    strncpy(shelf.name, name, sizeof(shelf.name) - 1);
    shelf.name[sizeof(shelf.name) - 1] = '\0';
    shelf.capacity = capacity;
    shelf.books = (Book *)malloc(sizeof(Book) * capacity);
    shelf.count = 0;
    return shelf;
}

void free_shelf(Shelf *shelf) {
    free(shelf->books);
    shelf->books = NULL;
    shelf->count = 0;
}
