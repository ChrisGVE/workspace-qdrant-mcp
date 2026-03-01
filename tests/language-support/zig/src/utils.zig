const std = @import("std");
const models = @import("models.zig");
const Book = models.Book;

pub fn validateIsbn(isbn: []const u8) bool {
    if (isbn.len != 13) return false;
    var total: i32 = 0;
    for (isbn, 0..) |ch, i| {
        const d: i32 = @as(i32, ch) - '0';
        if (d < 0 or d > 9) return false;
        if (i % 2 == 0) {
            total += d;
        } else {
            total += d * 3;
        }
    }
    return @mod(total, 10) == 0;
}

pub fn formatBook(buf: []u8, b: Book) []const u8 {
    const result = std.fmt.bufPrint(buf, "\"{s}\" by {s} ({d}) [ISBN: {s}]", .{
        b.title, b.author, b.year, b.isbn,
    }) catch return "";
    return result;
}

pub fn parseCsvLine(line: []const u8) ?Book {
    var fields: [5][]const u8 = undefined;
    var field_count: usize = 0;
    var start: usize = 0;
    for (line, 0..) |ch, i| {
        if (ch == ',') {
            if (field_count >= 5) return null;
            fields[field_count] = line[start..i];
            field_count += 1;
            start = i + 1;
        }
    }
    if (field_count >= 5) return null;
    fields[field_count] = line[start..];
    field_count += 1;
    if (field_count != 5) return null;

    const year = std.fmt.parseInt(i32, std.mem.trim(u8, fields[2], " "), 10) catch return null;

    const avail_str = std.mem.trim(u8, fields[4], " ");
    var available: bool = undefined;
    if (std.mem.eql(u8, avail_str, "true")) {
        available = true;
    } else if (std.mem.eql(u8, avail_str, "false")) {
        available = false;
    } else {
        return null;
    }

    return Book{
        .title = std.mem.trim(u8, fields[0], " "),
        .author = std.mem.trim(u8, fields[1], " "),
        .year = year,
        .isbn = std.mem.trim(u8, fields[3], " "),
        .available = available,
    };
}
