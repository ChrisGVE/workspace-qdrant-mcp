const std = @import("std");
const models = @import("models.zig");
const Book = models.Book;
const Shelf = models.Shelf;

pub fn addBook(allocator: std.mem.Allocator, s: *Shelf, b: Book) !void {
    if (isFull(s.*)) return error.ShelfFull;
    try s.books.append(allocator, b);
}

pub fn removeBook(allocator: std.mem.Allocator, s: *Shelf, isbn: []const u8) !Book {
    _ = allocator;
    for (s.books.items, 0..) |b, i| {
        if (std.mem.eql(u8, b.isbn, isbn)) {
            return s.books.orderedRemove(i);
        }
    }
    return error.NotFound;
}

pub fn findByAuthor(allocator: std.mem.Allocator, s: Shelf, author: []const u8) !std.ArrayList(Book) {
    var result = std.ArrayList(Book).empty;
    var lower_query_buf: [256]u8 = undefined;
    const query_len = @min(author.len, 256);
    for (author[0..query_len], 0..) |ch, i| {
        lower_query_buf[i] = std.ascii.toLower(ch);
    }
    const query = lower_query_buf[0..query_len];
    for (s.books.items) |b| {
        var lower_author_buf: [256]u8 = undefined;
        const a_len = @min(b.author.len, 256);
        for (b.author[0..a_len], 0..) |ch, i| {
            lower_author_buf[i] = std.ascii.toLower(ch);
        }
        if (std.mem.indexOf(u8, lower_author_buf[0..a_len], query) != null) {
            try result.append(allocator, b);
        }
    }
    return result;
}

pub fn findByYearRange(allocator: std.mem.Allocator, s: Shelf, start_year: i32, end_year: i32) !std.ArrayList(Book) {
    var result = std.ArrayList(Book).empty;
    for (s.books.items) |b| {
        if (b.year >= start_year and b.year <= end_year) {
            try result.append(allocator, b);
        }
    }
    return result;
}

pub fn isFull(s: Shelf) bool {
    return s.books.items.len >= s.capacity;
}

pub fn generateReport(buf: []u8, s: Shelf) []const u8 {
    const total: usize = s.books.items.len;
    var avail_count: usize = 0;
    var min_year: i32 = 9999;
    var max_year: i32 = 0;

    const AuthorEntry = struct { name: []const u8, count: usize };
    var authors: [100]AuthorEntry = undefined;
    var num_authors: usize = 0;

    for (s.books.items) |b| {
        if (b.available) avail_count += 1;
        if (b.year < min_year) min_year = b.year;
        if (b.year > max_year) max_year = b.year;
        var found = false;
        for (authors[0..num_authors]) |*a| {
            if (std.mem.eql(u8, a.name, b.author)) {
                a.count += 1;
                found = true;
                break;
            }
        }
        if (!found) {
            authors[num_authors] = AuthorEntry{ .name = b.author, .count = 1 };
            num_authors += 1;
        }
    }

    const avail_pct: usize = if (total > 0) avail_count * 100 / total else 0;
    const cap_pct: usize = if (s.capacity > 0) total * 100 / s.capacity else 0;

    var stream = std.io.fixedBufferStream(buf);
    const w = stream.writer();

    w.print("=== Library Report: {s} ===\n", .{s.name}) catch {};
    w.print("Total books: {d}\n", .{total}) catch {};
    w.print("Available: {d} / {d} ({d}%)\n", .{ avail_count, total, avail_pct }) catch {};
    w.print("Capacity: {d} / {d} ({d}% full)\n", .{ total, s.capacity, cap_pct }) catch {};
    w.print("\n", .{}) catch {};
    w.print("Authors ({d} unique):\n", .{num_authors}) catch {};
    for (authors[0..num_authors]) |a| {
        w.print("  - {s} ({d} books)\n", .{ a.name, a.count }) catch {};
    }
    w.print("\n", .{}) catch {};
    w.print("Year range: {d} - {d}\n", .{ min_year, max_year }) catch {};
    w.print("\n", .{}) catch {};
    w.print("Books by availability:\n", .{}) catch {};
    for (s.books.items) |b| {
        const marker: []const u8 = if (b.available) "+" else "-";
        w.print("  [{s}] {s} by {s} ({d}) - {s}\n", .{
            marker, b.title, b.author, b.year, b.isbn,
        }) catch {};
    }

    return stream.getWritten();
}
