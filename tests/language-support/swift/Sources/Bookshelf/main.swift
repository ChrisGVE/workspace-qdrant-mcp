var shelf = Shelf(name: "Computer Science", capacity: 10)

let books = [
    Book(title: "The Art of Computer Programming",
         author: "Donald Knuth", year: 1968,
         isbn: "9780201896831", available: true),
    Book(title: "Structure and Interpretation of Computer Programs",
         author: "Harold Abelson", year: 1996,
         isbn: "9780262510875", available: true),
    Book(title: "Introduction to Algorithms",
         author: "Thomas Cormen", year: 2009,
         isbn: "9780262033848", available: false),
    Book(title: "Design Patterns",
         author: "Erich Gamma", year: 1994,
         isbn: "9780201633610", available: true),
    Book(title: "The Pragmatic Programmer",
         author: "David Thomas", year: 2019,
         isbn: "9780135957059", available: true),
]

for book in books {
    try! addBook(&shelf, book)
}

print(generateReport(shelf))

print()
print("--- Search by author \"knuth\" ---")
for book in findByAuthor(shelf, author: "knuth") {
    print("  \(formatBook(book))")
}
print()

print("--- Search by year range 1990-2010 ---")
for book in findByYearRange(shelf, startYear: 1990, endYear: 2010) {
    print("  \(formatBook(book))")
}
print()

print("--- Parse CSV ---")
let parsed = parseCsvLine(
    "Clean Code,Robert Martin,2008,9780132350884,true")
print("  Parsed: \(formatBook(parsed))")
print()

print("--- ISBN Validation ---")
for book in books {
    let status = validateIsbn(book.isbn) ? "valid" : "invalid"
    print("  \(book.isbn): \(status)")
}
