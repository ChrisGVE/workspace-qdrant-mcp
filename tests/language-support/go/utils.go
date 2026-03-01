package main

import (
	"fmt"
	"strconv"
	"strings"
)

// ValidateIsbn checks whether isbn is a valid ISBN-13 number using the
// standard check-digit algorithm: alternate weights of 1 and 3, sum mod 10 == 0.
func ValidateIsbn(isbn string) bool {
	if len(isbn) != 13 {
		return false
	}
	sum := 0
	for i, ch := range isbn {
		if ch < '0' || ch > '9' {
			return false
		}
		digit := int(ch - '0')
		if i%2 == 0 {
			sum += digit
		} else {
			sum += digit * 3
		}
	}
	return sum%10 == 0
}

// FormatBook returns a single-line human-readable description of a book.
func FormatBook(book Book) string {
	return fmt.Sprintf("%q by %s (%d) [ISBN: %s]", book.Title, book.Author, book.Year, book.Isbn)
}

// ParseCsvLine parses a line in the format "title,author,year,isbn,available"
// and returns the corresponding Book or an error on malformed input.
func ParseCsvLine(line string) (Book, error) {
	fields := strings.Split(line, ",")
	if len(fields) != 5 {
		return Book{}, fmt.Errorf("expected 5 fields, got %d", len(fields))
	}
	title := strings.TrimSpace(fields[0])
	author := strings.TrimSpace(fields[1])
	yearStr := strings.TrimSpace(fields[2])
	isbn := strings.TrimSpace(fields[3])
	availStr := strings.TrimSpace(fields[4])

	year, err := strconv.Atoi(yearStr)
	if err != nil {
		return Book{}, fmt.Errorf("invalid year %q: %w", yearStr, err)
	}

	var available bool
	switch availStr {
	case "true":
		available = true
	case "false":
		available = false
	default:
		return Book{}, fmt.Errorf("invalid available value %q: must be true or false", availStr)
	}

	return NewBook(title, author, year, isbn, available), nil
}
