local Models = require("models")
local Storage = require("storage")
local Utils = require("utils")

local function main()
    local shelf = Models.new_shelf("Computer Science", 10)

    local books = {
        Models.new_book("The Art of Computer Programming", "Donald Knuth", 1968, "9780201896831", true),
        Models.new_book("Structure and Interpretation of Computer Programs", "Harold Abelson", 1996, "9780262510875", true),
        Models.new_book("Introduction to Algorithms", "Thomas Cormen", 2009, "9780262033848", false),
        Models.new_book("Design Patterns", "Erich Gamma", 1994, "9780201633610", true),
        Models.new_book("The Pragmatic Programmer", "David Thomas", 2019, "9780135957059", true),
    }
    for _, book in ipairs(books) do
        Storage.add_book(shelf, book)
    end

    print(Storage.generate_report(shelf))
    print()

    print('--- Search by author "knuth" ---')
    for _, book in ipairs(Storage.find_by_author(shelf, "knuth")) do
        print("  " .. Utils.format_book(book))
    end
    print()

    print("--- Search by year range 1990-2010 ---")
    for _, book in ipairs(Storage.find_by_year_range(shelf, 1990, 2010)) do
        print("  " .. Utils.format_book(book))
    end
    print()

    print("--- Parse CSV ---")
    local parsed = Utils.parse_csv_line("Clean Code,Robert Martin,2008,9780132350884,true")
    print("  Parsed: " .. Utils.format_book(parsed))
    print()

    print("--- ISBN Validation ---")
    for _, book in ipairs(books) do
        local status = Utils.validate_isbn(book.isbn) and "valid" or "invalid"
        print("  " .. book.isbn .. ": " .. status)
    end
end

main()
