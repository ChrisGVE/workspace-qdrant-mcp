import Foundation

enum ParseError: Error {
    case wrongFieldCount(Int)
    case invalidYear(String)
    case invalidBoolean(String)
}

func validateIsbn(_ isbn: String) -> Bool {
    let chars = Array(isbn)
    if chars.count != 13 { return false }
    for c in chars {
        if !c.isNumber { return false }
    }
    var total = 0
    for (i, c) in chars.enumerated() {
        let digit = c.wholeNumberValue!
        total += (i % 2 == 0) ? digit : digit * 3
    }
    return total % 10 == 0
}

func formatBook(_ book: Book) -> String {
    return "\"\(book.title)\" by \(book.author)"
        + " (\(book.year)) [ISBN: \(book.isbn)]"
}

func parseCsvLine(_ line: String) -> Book {
    let parts = line.split(separator: ",", omittingEmptySubsequences: false)
        .map { String($0) }
    guard parts.count == 5 else {
        fatalError("Expected 5 fields, got \(parts.count)")
    }

    guard let year = Int(parts[2].trimmingCharacters(in: .whitespaces))
    else {
        fatalError("Invalid year: \(parts[2])")
    }

    let availStr = parts[4].trimmingCharacters(in: .whitespaces)
        .lowercased()
    let available: Bool
    if availStr == "true" {
        available = true
    } else if availStr == "false" {
        available = false
    } else {
        fatalError("Invalid boolean: \(parts[4])")
    }

    return Book(
        title: parts[0].trimmingCharacters(in: .whitespaces),
        author: parts[1].trimmingCharacters(in: .whitespaces),
        year: year,
        isbn: parts[3].trimmingCharacters(in: .whitespaces),
        available: available
    )
}
