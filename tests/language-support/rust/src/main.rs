mod models;
mod storage;
mod utils;

use models::{Book, Shelf};

fn main() {
    let mut shelf = Shelf::new("Computer Science", 10);

    let books = vec![
        Book::new("The Art of Computer Programming", "Donald Knuth", 1968, "9780201896831", true),
        Book::new("Structure and Interpretation of Computer Programs", "Harold Abelson", 1996, "9780262510875", true),
        Book::new("Introduction to Algorithms", "Thomas Cormen", 2009, "9780262033848", false),
        Book::new("Design Patterns", "Erich Gamma", 1994, "9780201633610", true),
        Book::new("The Pragmatic Programmer", "David Thomas", 2019, "9780135957059", true),
    ];

    for book in books {
        storage::add_book(&mut shelf, book).expect("Failed to add book");
    }

    let report = storage::generate_report(&shelf);
    print!("{}", report);

    println!("\n--- Search by author \"knuth\" ---");
    for book in storage::find_by_author(&shelf, "knuth") {
        println!("  {}", utils::format_book(&book));
    }

    println!("\n--- Search by year range 1990-2010 ---");
    for book in storage::find_by_year_range(&shelf, 1990, 2010) {
        println!("  {}", utils::format_book(&book));
    }

    println!("\n--- Parse CSV ---");
    match utils::parse_csv_line("Clean Code,Robert Martin,2008,9780132350884,true") {
        Ok(book) => println!("  Parsed: {}", utils::format_book(&book)),
        Err(e) => println!("  Error: {}", e),
    }

    println!("\n--- ISBN Validation ---");
    for book in &shelf.books {
        let status = if utils::validate_isbn(&book.isbn) {
            "valid"
        } else {
            "invalid"
        };
        println!("  {}: {}", book.isbn, status);
    }
}
