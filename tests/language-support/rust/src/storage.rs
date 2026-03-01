use std::fmt::Write;

use crate::models::{Book, Shelf};

pub fn add_book(shelf: &mut Shelf, book: Book) -> Result<(), String> {
    if shelf.books.len() >= shelf.capacity {
        return Err(format!("Shelf \"{}\" is at capacity", shelf.name));
    }
    shelf.books.push(book);
    Ok(())
}

pub fn remove_book(shelf: &mut Shelf, isbn: &str) -> Result<Book, String> {
    let pos = shelf
        .books
        .iter()
        .position(|b| b.isbn == isbn)
        .ok_or_else(|| format!("Book with ISBN {} not found", isbn))?;
    Ok(shelf.books.remove(pos))
}

pub fn find_by_author(shelf: &Shelf, author: &str) -> Vec<Book> {
    let query = author.to_lowercase();
    shelf
        .books
        .iter()
        .filter(|b| b.author.to_lowercase().contains(&query))
        .cloned()
        .collect()
}

pub fn find_by_year_range(shelf: &Shelf, start: i32, end: i32) -> Vec<Book> {
    shelf
        .books
        .iter()
        .filter(|b| b.year >= start && b.year <= end)
        .cloned()
        .collect()
}

pub fn sort_by_title(shelf: &Shelf) -> Vec<Book> {
    let mut sorted: Vec<Book> = shelf.books.clone();
    sorted.sort_by(|a, b| a.title.cmp(&b.title));
    sorted
}

pub fn is_full(shelf: &Shelf) -> bool {
    shelf.books.len() >= shelf.capacity
}

pub fn generate_report(shelf: &Shelf) -> String {
    let total = shelf.books.len();
    let avail = shelf.books.iter().filter(|b| b.available).count();
    let avail_pct = if total > 0 { avail * 100 / total } else { 0 };
    let cap_pct = if shelf.capacity > 0 { total * 100 / shelf.capacity } else { 0 };
    let mut authors: Vec<(String, usize)> = Vec::new();
    for book in &shelf.books {
        if let Some(e) = authors.iter_mut().find(|(a, _)| *a == book.author) {
            e.1 += 1;
        } else {
            authors.push((book.author.clone(), 1));
        }
    }
    let min_year = shelf.books.iter().map(|b| b.year).min().unwrap_or(0);
    let max_year = shelf.books.iter().map(|b| b.year).max().unwrap_or(0);
    let mut r = String::new();
    writeln!(r, "=== Library Report: {} ===", shelf.name).unwrap();
    writeln!(r, "Total books: {}", total).unwrap();
    writeln!(r, "Available: {} / {} ({}%)", avail, total, avail_pct).unwrap();
    writeln!(r, "Capacity: {} / {} ({}% full)", total, shelf.capacity, cap_pct).unwrap();
    writeln!(r).unwrap();
    writeln!(r, "Authors ({} unique):", authors.len()).unwrap();
    for (author, count) in &authors {
        writeln!(r, "  - {} ({} books)", author, count).unwrap();
    }
    writeln!(r).unwrap();
    writeln!(r, "Year range: {} - {}", min_year, max_year).unwrap();
    writeln!(r).unwrap();
    writeln!(r, "Books by availability:").unwrap();
    for book in &shelf.books {
        let m = if book.available { "+" } else { "-" };
        writeln!(r, "  [{}] {} by {} ({}) - {}", m, book.title, book.author, book.year, book.isbn).unwrap();
    }
    r
}
