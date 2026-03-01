package main

import (
	"fmt"
	"log"
)

func main() {
	shelf := NewShelf("Computer Science", 10)

	books := []Book{
		NewBook("The Art of Computer Programming", "Donald Knuth", 1968, "9780201896831", true),
		NewBook("Structure and Interpretation of Computer Programs", "Harold Abelson", 1996, "9780262510875", true),
		NewBook("Introduction to Algorithms", "Thomas Cormen", 2009, "9780262033848", false),
		NewBook("Design Patterns", "Erich Gamma", 1994, "9780201633610", true),
		NewBook("The Pragmatic Programmer", "David Thomas", 2019, "9780135957059", true),
	}

	for _, b := range books {
		if err := AddBook(&shelf, b); err != nil {
			log.Fatalf("AddBook: %v", err)
		}
	}

	fmt.Print(GenerateReport(&shelf))

	fmt.Println()
	fmt.Println(`--- Search by author "knuth" ---`)
	for _, b := range FindByAuthor(&shelf, "knuth") {
		fmt.Printf("  %s\n", FormatBook(b))
	}

	fmt.Println()
	fmt.Println("--- Search by year range 1990-2010 ---")
	for _, b := range FindByYearRange(&shelf, 1990, 2010) {
		fmt.Printf("  %s\n", FormatBook(b))
	}

	fmt.Println()
	fmt.Println("--- Parse CSV ---")
	parsed, err := ParseCsvLine("Clean Code,Robert Martin,2008,9780132350884,true")
	if err != nil {
		log.Fatalf("ParseCsvLine: %v", err)
	}
	fmt.Printf("  Parsed: %s\n", FormatBook(parsed))

	fmt.Println()
	fmt.Println("--- ISBN Validation ---")
	for _, b := range shelf.Books {
		status := "valid"
		if !ValidateIsbn(b.Isbn) {
			status = "invalid"
		}
		fmt.Printf("  %s: %s\n", b.Isbn, status)
	}
}
