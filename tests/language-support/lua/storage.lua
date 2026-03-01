local Storage = {}

function Storage.add_book(shelf, book)
    if Storage.is_full(shelf) then
        error("Shelf '" .. shelf.name .. "' is at capacity (" .. shelf.capacity .. ")")
    end
    table.insert(shelf.books, book)
end

function Storage.remove_book(shelf, isbn)
    for i, book in ipairs(shelf.books) do
        if book.isbn == isbn then
            return table.remove(shelf.books, i)
        end
    end
    error("No book with ISBN '" .. isbn .. "' found on shelf")
end

function Storage.find_by_author(shelf, author)
    local query = string.lower(author)
    local results = {}
    for _, book in ipairs(shelf.books) do
        if string.find(string.lower(book.author), query, 1, true) then
            table.insert(results, book)
        end
    end
    return results
end

function Storage.find_by_year_range(shelf, start_year, end_year)
    local results = {}
    for _, book in ipairs(shelf.books) do
        if book.year >= start_year and book.year <= end_year then
            table.insert(results, book)
        end
    end
    return results
end

function Storage.sort_by_title(shelf)
    local sorted = {}
    for _, book in ipairs(shelf.books) do
        table.insert(sorted, book)
    end
    table.sort(sorted, function(a, b) return a.title < b.title end)
    return sorted
end

function Storage.is_full(shelf)
    return #shelf.books >= shelf.capacity
end

function Storage.generate_report(shelf)
    local books = shelf.books
    local total = #books
    local available_count = 0
    for _, book in ipairs(books) do
        if book.available then available_count = available_count + 1 end
    end
    local avail_pct = total > 0 and math.floor(available_count * 100 / total) or 0
    local cap_pct = shelf.capacity > 0 and math.floor(total * 100 / shelf.capacity) or 0

    local author_counts = {}
    local author_order = {}
    for _, book in ipairs(books) do
        if not author_counts[book.author] then
            table.insert(author_order, book.author)
            author_counts[book.author] = 0
        end
        author_counts[book.author] = author_counts[book.author] + 1
    end
    local unique_authors = #author_order

    local min_year, max_year = 0, 0
    if total > 0 then
        min_year = books[1].year
        max_year = books[1].year
        for _, book in ipairs(books) do
            if book.year < min_year then min_year = book.year end
            if book.year > max_year then max_year = book.year end
        end
    end

    local lines = {}
    table.insert(lines, "=== Library Report: " .. shelf.name .. " ===")
    table.insert(lines, "Total books: " .. total)
    table.insert(lines, "Available: " .. available_count .. " / " .. total .. " (" .. avail_pct .. "%)")
    table.insert(lines, "Capacity: " .. total .. " / " .. shelf.capacity .. " (" .. cap_pct .. "% full)")
    table.insert(lines, "")

    table.insert(lines, "Authors (" .. unique_authors .. " unique):")
    for _, author in ipairs(author_order) do
        table.insert(lines, "  - " .. author .. " (" .. author_counts[author] .. " books)")
    end
    table.insert(lines, "")

    table.insert(lines, "Year range: " .. min_year .. " - " .. max_year)
    table.insert(lines, "")

    table.insert(lines, "Books by availability:")
    for _, book in ipairs(books) do
        local marker = book.available and "+" or "-"
        table.insert(lines, "  [" .. marker .. "] " .. book.title .. " by " .. book.author .. " (" .. book.year .. ") - " .. book.isbn)
    end

    return table.concat(lines, "\n")
end

return Storage
