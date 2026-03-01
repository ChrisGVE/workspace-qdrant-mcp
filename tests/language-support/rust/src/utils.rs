use crate::models::Book;

pub fn validate_isbn(isbn: &str) -> bool {
    let digits: Vec<u32> = isbn.chars().filter_map(|c| c.to_digit(10)).collect();
    if digits.len() != 13 {
        return false;
    }
    let sum: u32 = digits
        .iter()
        .enumerate()
        .map(|(i, &d)| if i % 2 == 0 { d } else { d * 3 })
        .sum();
    sum % 10 == 0
}

pub fn format_book(book: &Book) -> String {
    format!(
        "\"{}\" by {} ({}) [ISBN: {}]",
        book.title, book.author, book.year, book.isbn
    )
}

pub fn parse_csv_line(line: &str) -> Result<Book, String> {
    let fields: Vec<&str> = line.split(',').collect();
    if fields.len() != 5 {
        return Err(format!(
            "Expected 5 fields, got {}",
            fields.len()
        ));
    }
    let year: i32 = fields[2]
        .trim()
        .parse()
        .map_err(|_| format!("Invalid year: {}", fields[2].trim()))?;
    let available: bool = fields[4]
        .trim()
        .parse()
        .map_err(|_| format!("Invalid boolean: {}", fields[4].trim()))?;
    Ok(Book::new(
        fields[0].trim(),
        fields[1].trim(),
        year,
        fields[3].trim(),
        available,
    ))
}
