package main

// Genre represents the category of a book.
type Genre int

const (
	Fiction Genre = iota
	NonFiction
	Science
	History
	Biography
	Technology
	Philosophy
)

// Book holds metadata for a single library book.
type Book struct {
	Title     string
	Author    string
	Year      int
	Isbn      string
	Available bool
}

// Shelf is a named collection of books with a fixed capacity.
type Shelf struct {
	Name     string
	Capacity int
	Books    []Book
}

// NewBook creates a Book with the given fields.
func NewBook(title, author string, year int, isbn string, available bool) Book {
	return Book{
		Title:     title,
		Author:    author,
		Year:      year,
		Isbn:      isbn,
		Available: available,
	}
}

// NewShelf creates an empty Shelf with the given name and capacity.
func NewShelf(name string, capacity int) Shelf {
	return Shelf{
		Name:     name,
		Capacity: capacity,
		Books:    []Book{},
	}
}
