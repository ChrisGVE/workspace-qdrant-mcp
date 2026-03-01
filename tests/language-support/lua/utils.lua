local Models = require("models")

local Utils = {}

function Utils.validate_isbn(isbn)
    if type(isbn) ~= "string" or #isbn ~= 13 or not isbn:match("^%d+$") then
        return false
    end
    local total = 0
    for i = 1, 13 do
        local digit = tonumber(isbn:sub(i, i))
        local weight = (i % 2 == 1) and 1 or 3
        total = total + digit * weight
    end
    return (total % 10) == 0
end

function Utils.format_book(book)
    return '"' .. book.title .. '" by ' .. book.author ..
           ' (' .. book.year .. ') [ISBN: ' .. book.isbn .. ']'
end

function Utils.parse_csv_line(line)
    local parts = {}
    for field in line:gmatch("([^,]+)") do
        table.insert(parts, field)
    end
    if #parts ~= 5 then
        error("Expected 5 fields, got " .. #parts)
    end

    local title = parts[1]:match("^%s*(.-)%s*$")
    local author = parts[2]:match("^%s*(.-)%s*$")
    local year_str = parts[3]:match("^%s*(.-)%s*$")
    local isbn = parts[4]:match("^%s*(.-)%s*$")
    local avail_str = parts[5]:match("^%s*(.-)%s*$")

    local year = tonumber(year_str)
    if not year then
        error("Invalid year: " .. year_str)
    end

    local available
    local avail_lower = avail_str:lower()
    if avail_lower == "true" then
        available = true
    elseif avail_lower == "false" then
        available = false
    else
        error("Invalid boolean: " .. avail_str)
    end

    return Models.new_book(title, author, year, isbn, available)
end

return Utils
