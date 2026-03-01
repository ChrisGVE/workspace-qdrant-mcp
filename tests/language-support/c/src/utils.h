#ifndef UTILS_H
#define UTILS_H

#include "models.h"

int validate_isbn(const char *isbn);
void format_book(const Book *book, char *buffer, int buf_size);
int parse_csv_line(const char *line, Book *result);

#endif
