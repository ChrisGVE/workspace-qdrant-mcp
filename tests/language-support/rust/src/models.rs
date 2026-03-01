use std::fmt;

#[derive(Clone)]
pub struct Book {
    pub title: String,
    pub author: String,
    pub year: i32,
    pub isbn: String,
    pub available: bool,
}

impl Book {
    pub fn new(title: &str, author: &str, year: i32, isbn: &str, available: bool) -> Self {
        Self {
            title: title.to_string(),
            author: author.to_string(),
            year,
            isbn: isbn.to_string(),
            available,
        }
    }
}

impl fmt::Display for Book {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "\"{}\" by {} ({}) [ISBN: {}]",
            self.title, self.author, self.year, self.isbn
        )
    }
}

pub struct Shelf {
    pub name: String,
    pub capacity: usize,
    pub books: Vec<Book>,
}

impl Shelf {
    pub fn new(name: &str, capacity: usize) -> Self {
        Self {
            name: name.to_string(),
            capacity,
            books: Vec::new(),
        }
    }
}

pub enum Genre {
    Fiction,
    NonFiction,
    Science,
    History,
    Biography,
    Technology,
    Philosophy,
}
