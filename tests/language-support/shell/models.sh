#!/usr/bin/env bash
# Models: Book and Shelf data structures for Bash
# Books are stored as colon-separated strings:
#   title:author:year:isbn:available
# Shelf is stored as global variables.

# Genre constants
readonly GENRE_FICTION="Fiction"
readonly GENRE_NON_FICTION="NonFiction"
readonly GENRE_SCIENCE="Science"
readonly GENRE_HISTORY="History"
readonly GENRE_BIOGRAPHY="Biography"
readonly GENRE_TECHNOLOGY="Technology"
readonly GENRE_PHILOSOPHY="Philosophy"

# Create a book string from fields
make_book() {
    local title="$1" author="$2" year="$3" isbn="$4" available="$5"
    echo "${title}:${author}:${year}:${isbn}:${available}"
}

# Field accessors for a book string
book_title()     { echo "$1" | cut -d: -f1; }
book_author()    { echo "$1" | cut -d: -f2; }
book_year()      { echo "$1" | cut -d: -f3; }
book_isbn()      { echo "$1" | cut -d: -f4; }
book_available() { echo "$1" | cut -d: -f5; }

# Shelf globals
SHELF_NAME=""
SHELF_CAPACITY=0
SHELF_BOOKS=()

# Initialize the shelf
init_shelf() {
    SHELF_NAME="$1"
    SHELF_CAPACITY="$2"
    SHELF_BOOKS=()
}
