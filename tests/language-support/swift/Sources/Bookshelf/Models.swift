enum Genre {
    case fiction
    case nonFiction
    case science
    case history
    case biography
    case technology
    case philosophy
}

struct Book {
    let title: String
    let author: String
    let year: Int
    let isbn: String
    let available: Bool
}

struct Shelf {
    let name: String
    let capacity: Int
    var books: [Book]

    init(name: String, capacity: Int) {
        self.name = name
        self.capacity = capacity
        self.books = []
    }
}
