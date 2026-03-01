#include "utils.h"
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int validate_isbn(const char *isbn) {
    if (strlen(isbn) != 13) return 0;
    for (int i = 0; i < 13; i++) {
        if (!isdigit((unsigned char)isbn[i])) return 0;
    }
    int total = 0;
    for (int i = 0; i < 13; i++) {
        int digit = isbn[i] - '0';
        total += (i % 2 == 0) ? digit : digit * 3;
    }
    return total % 10 == 0;
}

void format_book(const Book *book, char *buffer, int buf_size) {
    snprintf(buffer, buf_size, "\"%s\" by %s (%d) [ISBN: %s]",
             book->title, book->author, book->year, book->isbn);
}

int parse_csv_line(const char *line, Book *result) {
    char buf[512];
    strncpy(buf, line, sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = '\0';

    char *fields[5];
    int field_count = 0;
    char *token = strtok(buf, ",");
    while (token != NULL && field_count < 5) {
        fields[field_count++] = token;
        token = strtok(NULL, ",");
    }
    if (field_count != 5) return -1;

    char *endptr;
    long year = strtol(fields[2], &endptr, 10);
    if (*endptr != '\0') return -1;

    /* Trim whitespace from available field */
    char *avail = fields[4];
    while (*avail == ' ') avail++;
    int len = strlen(avail);
    while (len > 0 && avail[len - 1] == ' ') { avail[--len] = '\0'; }

    bool available;
    if (strcmp(avail, "true") == 0) {
        available = true;
    } else if (strcmp(avail, "false") == 0) {
        available = false;
    } else {
        return -1;
    }

    *result = create_book(fields[0], fields[1], (int)year,
                          fields[3], available);
    return 0;
}
