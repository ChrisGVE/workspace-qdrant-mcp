#ifndef STORAGE_H
#define STORAGE_H

#include "models.h"

int add_book(Shelf *shelf, Book book);
int remove_book(Shelf *shelf, const char *isbn, Book *removed);
int find_by_author(const Shelf *shelf, const char *author,
                   Book *results, int max_results);
int find_by_year_range(const Shelf *shelf, int start_year, int end_year,
                       Book *results, int max_results);
void sort_by_title(const Shelf *shelf, Book *sorted, int *count);
int is_full(const Shelf *shelf);
void generate_report(const Shelf *shelf, char *buffer, int buf_size);

#endif
