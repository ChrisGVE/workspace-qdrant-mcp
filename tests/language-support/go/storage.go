package main

import (
	"fmt"
	"sort"
	"strings"
)

// AddBook adds a book to the shelf, returning an error if the shelf is full.
func AddBook(shelf *Shelf, book Book) error {
	if IsFull(shelf) {
		return fmt.Errorf("shelf %q is at capacity (%d)", shelf.Name, shelf.Capacity)
	}
	shelf.Books = append(shelf.Books, book)
	return nil
}

// RemoveBook removes the book with the given ISBN, returning an error if not found.
func RemoveBook(shelf *Shelf, isbn string) error {
	for i, b := range shelf.Books {
		if b.Isbn == isbn {
			shelf.Books = append(shelf.Books[:i], shelf.Books[i+1:]...)
			return nil
		}
	}
	return fmt.Errorf("book with ISBN %q not found", isbn)
}

// FindByAuthor returns all books whose author contains the query (case-insensitive).
func FindByAuthor(shelf *Shelf, author string) []Book {
	query := strings.ToLower(author)
	var result []Book
	for _, b := range shelf.Books {
		if strings.Contains(strings.ToLower(b.Author), query) {
			result = append(result, b)
		}
	}
	return result
}

// FindByYearRange returns all books published within [start, end] inclusive.
func FindByYearRange(shelf *Shelf, start, end int) []Book {
	var result []Book
	for _, b := range shelf.Books {
		if b.Year >= start && b.Year <= end {
			result = append(result, b)
		}
	}
	return result
}

// SortByTitle returns a copy of the shelf's books sorted alphabetically by title.
func SortByTitle(shelf *Shelf) []Book {
	result := make([]Book, len(shelf.Books))
	copy(result, shelf.Books)
	sort.Slice(result, func(i, j int) bool {
		return result[i].Title < result[j].Title
	})
	return result
}

// IsFull reports whether the shelf has reached its capacity.
func IsFull(shelf *Shelf) bool {
	return len(shelf.Books) >= shelf.Capacity
}

// GenerateReport produces a multi-line formatted summary of the shelf.
func GenerateReport(shelf *Shelf) string {
	total, available := len(shelf.Books), 0
	for _, b := range shelf.Books {
		if b.Available {
			available++
		}
	}
	availPct, capacityPct := 0, 0
	if total > 0 {
		availPct = available * 100 / total
		capacityPct = total * 100 / shelf.Capacity
	}
	// Collect author counts preserving insertion order for display.
	authorOrder, authorCount := []string{}, map[string]int{}
	minYear, maxYear := shelf.Books[0].Year, shelf.Books[0].Year
	for _, b := range shelf.Books {
		if _, seen := authorCount[b.Author]; !seen {
			authorOrder = append(authorOrder, b.Author)
		}
		authorCount[b.Author]++
		if b.Year < minYear {
			minYear = b.Year
		}
		if b.Year > maxYear {
			maxYear = b.Year
		}
	}
	var sb strings.Builder
	fmt.Fprintf(&sb, "=== Library Report: %s ===\n", shelf.Name)
	fmt.Fprintf(&sb, "Total books: %d\n", total)
	fmt.Fprintf(&sb, "Available: %d / %d (%d%%)\n", available, total, availPct)
	fmt.Fprintf(&sb, "Capacity: %d / %d (%d%% full)\n", total, shelf.Capacity, capacityPct)
	fmt.Fprintf(&sb, "\nAuthors (%d unique):\n", len(authorOrder))
	for _, a := range authorOrder {
		fmt.Fprintf(&sb, "  - %s (%d books)\n", a, authorCount[a])
	}
	fmt.Fprintf(&sb, "\nYear range: %d - %d\n", minYear, maxYear)
	fmt.Fprintf(&sb, "\nBooks by availability:\n")
	for _, b := range shelf.Books {
		mark := "+"
		if !b.Available {
			mark = "-"
		}
		fmt.Fprintf(&sb, "  [%s] %s by %s (%d) - %s\n", mark, b.Title, b.Author, b.Year, b.Isbn)
	}
	return sb.String()
}
