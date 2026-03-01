#ifndef MODELS_H
#define MODELS_H

#include <stdbool.h>

#define GENRE_FICTION     0
#define GENRE_NON_FICTION 1
#define GENRE_SCIENCE     2
#define GENRE_HISTORY     3
#define GENRE_BIOGRAPHY   4
#define GENRE_TECHNOLOGY  5
#define GENRE_PHILOSOPHY  6

typedef struct {
    char title[256];
    char author[128];
    int year;
    char isbn[14];
    bool available;
} Book;

typedef struct {
    char name[128];
    int capacity;
    Book *books;
    int count;
} Shelf;

Book create_book(const char *title, const char *author, int year,
                 const char *isbn, bool available);
Shelf create_shelf(const char *name, int capacity);
void free_shelf(Shelf *shelf);

#endif
