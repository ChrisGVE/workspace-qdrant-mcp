#include "storage.h"
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int add_book(Shelf *shelf, Book book) {
    if (is_full(shelf)) {
        return -1;
    }
    shelf->books[shelf->count] = book;
    shelf->count++;
    return 0;
}

int remove_book(Shelf *shelf, const char *isbn, Book *removed) {
    for (int i = 0; i < shelf->count; i++) {
        if (strcmp(shelf->books[i].isbn, isbn) == 0) {
            *removed = shelf->books[i];
            for (int j = i; j < shelf->count - 1; j++) {
                shelf->books[j] = shelf->books[j + 1];
            }
            shelf->count--;
            return 0;
        }
    }
    return -1;
}

static int str_contains_ci(const char *haystack, const char *needle) {
    int hlen = strlen(haystack);
    int nlen = strlen(needle);
    if (nlen > hlen) return 0;
    for (int i = 0; i <= hlen - nlen; i++) {
        int match = 1;
        for (int j = 0; j < nlen; j++) {
            if (tolower((unsigned char)haystack[i + j]) !=
                tolower((unsigned char)needle[j])) {
                match = 0;
                break;
            }
        }
        if (match) return 1;
    }
    return 0;
}

int find_by_author(const Shelf *shelf, const char *author,
                   Book *results, int max_results) {
    int found = 0;
    for (int i = 0; i < shelf->count && found < max_results; i++) {
        if (str_contains_ci(shelf->books[i].author, author)) {
            results[found] = shelf->books[i];
            found++;
        }
    }
    return found;
}

int find_by_year_range(const Shelf *shelf, int start_year, int end_year,
                       Book *results, int max_results) {
    int found = 0;
    for (int i = 0; i < shelf->count && found < max_results; i++) {
        if (shelf->books[i].year >= start_year &&
            shelf->books[i].year <= end_year) {
            results[found] = shelf->books[i];
            found++;
        }
    }
    return found;
}

static int cmp_title(const void *a, const void *b) {
    return strcmp(((const Book *)a)->title, ((const Book *)b)->title);
}

void sort_by_title(const Shelf *shelf, Book *sorted, int *count) {
    *count = shelf->count;
    memcpy(sorted, shelf->books, sizeof(Book) * shelf->count);
    qsort(sorted, *count, sizeof(Book), cmp_title);
}

int is_full(const Shelf *shelf) {
    return shelf->count >= shelf->capacity;
}

void generate_report(const Shelf *shelf, char *buffer, int buf_size) {
    int total = shelf->count;
    int available_count = 0;
    for (int i = 0; i < total; i++) {
        if (shelf->books[i].available) available_count++;
    }
    int avail_pct = total > 0 ? (available_count * 100 / total) : 0;
    int cap_pct = shelf->capacity > 0 ? (total * 100 / shelf->capacity) : 0;

    /* Count unique authors */
    char authors[64][128];
    int author_counts[64];
    int unique = 0;
    for (int i = 0; i < total; i++) {
        int found = -1;
        for (int j = 0; j < unique; j++) {
            if (strcmp(authors[j], shelf->books[i].author) == 0) {
                found = j;
                break;
            }
        }
        if (found >= 0) {
            author_counts[found]++;
        } else {
            strncpy(authors[unique], shelf->books[i].author, 127);
            authors[unique][127] = '\0';
            author_counts[unique] = 1;
            unique++;
        }
    }

    int min_year = shelf->books[0].year;
    int max_year = shelf->books[0].year;
    for (int i = 1; i < total; i++) {
        if (shelf->books[i].year < min_year) min_year = shelf->books[i].year;
        if (shelf->books[i].year > max_year) max_year = shelf->books[i].year;
    }

    int pos = 0;
    pos += snprintf(buffer + pos, buf_size - pos,
                    "=== Library Report: %s ===\n", shelf->name);
    pos += snprintf(buffer + pos, buf_size - pos,
                    "Total books: %d\n", total);
    pos += snprintf(buffer + pos, buf_size - pos,
                    "Available: %d / %d (%d%%)\n",
                    available_count, total, avail_pct);
    pos += snprintf(buffer + pos, buf_size - pos,
                    "Capacity: %d / %d (%d%% full)\n",
                    total, shelf->capacity, cap_pct);
    pos += snprintf(buffer + pos, buf_size - pos, "\n");
    pos += snprintf(buffer + pos, buf_size - pos,
                    "Authors (%d unique):\n", unique);
    for (int i = 0; i < unique; i++) {
        pos += snprintf(buffer + pos, buf_size - pos,
                        "  - %s (%d books)\n", authors[i], author_counts[i]);
    }
    pos += snprintf(buffer + pos, buf_size - pos, "\n");
    pos += snprintf(buffer + pos, buf_size - pos,
                    "Year range: %d - %d\n", min_year, max_year);
    pos += snprintf(buffer + pos, buf_size - pos, "\n");
    pos += snprintf(buffer + pos, buf_size - pos, "Books by availability:\n");
    for (int i = 0; i < total; i++) {
        char marker = shelf->books[i].available ? '+' : '-';
        pos += snprintf(buffer + pos, buf_size - pos,
                        "  [%c] %s by %s (%d) - %s\n",
                        marker, shelf->books[i].title,
                        shelf->books[i].author, shelf->books[i].year,
                        shelf->books[i].isbn);
    }
}
