import Foundation

enum StorageError: Error {
    case shelfFull(String)
    case bookNotFound(String)
}

func addBook(_ shelf: inout Shelf, _ book: Book) throws {
    if isFull(shelf) {
        throw StorageError.shelfFull(
            "Shelf '\(shelf.name)' is at capacity")
    }
    shelf.books.append(book)
}

func removeBook(_ shelf: inout Shelf, isbn: String) throws -> Book {
    guard let index = shelf.books.firstIndex(where: { $0.isbn == isbn })
    else {
        throw StorageError.bookNotFound(
            "No book with ISBN '\(isbn)' found")
    }
    return shelf.books.remove(at: index)
}

func findByAuthor(_ shelf: Shelf, author: String) -> [Book] {
    let query = author.lowercased()
    return shelf.books.filter {
        $0.author.lowercased().contains(query)
    }
}

func findByYearRange(_ shelf: Shelf, startYear: Int,
                     endYear: Int) -> [Book] {
    return shelf.books.filter {
        $0.year >= startYear && $0.year <= endYear
    }
}

func sortByTitle(_ shelf: Shelf) -> [Book] {
    return shelf.books.sorted { $0.title < $1.title }
}

func isFull(_ shelf: Shelf) -> Bool {
    return shelf.books.count >= shelf.capacity
}

func generateReport(_ shelf: Shelf) -> String {
    let total = shelf.books.count
    let availableCount = shelf.books.filter { $0.available }.count
    let availPct = total > 0 ? (availableCount * 100 / total) : 0
    let capPct = shelf.capacity > 0
        ? (total * 100 / shelf.capacity) : 0

    var authorOrder: [String] = []
    var authorCounts: [String: Int] = [:]
    for book in shelf.books {
        if authorCounts[book.author] == nil {
            authorOrder.append(book.author)
        }
        authorCounts[book.author, default: 0] += 1
    }

    let years = shelf.books.map { $0.year }
    let minYear = years.min() ?? 0
    let maxYear = years.max() ?? 0

    var lines: [String] = []
    lines.append("=== Library Report: \(shelf.name) ===")
    lines.append("Total books: \(total)")
    lines.append("Available: \(availableCount) / \(total) (\(availPct)%)")
    lines.append("Capacity: \(total) / \(shelf.capacity) (\(capPct)% full)")
    lines.append("")
    lines.append("Authors (\(authorOrder.count) unique):")
    for author in authorOrder {
        lines.append("  - \(author) (\(authorCounts[author]!) books)")
    }
    lines.append("")
    lines.append("Year range: \(minYear) - \(maxYear)")
    lines.append("")
    lines.append("Books by availability:")
    for book in shelf.books {
        let marker = book.available ? "+" : "-"
        lines.append(
            "  [\(marker)] \(book.title) by \(book.author)"
            + " (\(book.year)) - \(book.isbn)")
    }
    return lines.joined(separator: "\n")
}
