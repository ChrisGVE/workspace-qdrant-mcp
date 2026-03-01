local Models = {}

Models.Genre = {
    FICTION = "Fiction",
    NON_FICTION = "NonFiction",
    SCIENCE = "Science",
    HISTORY = "History",
    BIOGRAPHY = "Biography",
    TECHNOLOGY = "Technology",
    PHILOSOPHY = "Philosophy",
}

function Models.new_book(title, author, year, isbn, available)
    return {
        title = title,
        author = author,
        year = year,
        isbn = isbn,
        available = available,
    }
end

function Models.new_shelf(name, capacity)
    return {
        name = name,
        capacity = capacity,
        books = {},
    }
end

return Models
