const std = @import("std");
const models = @import("models.zig");
const storage = @import("storage.zig");
const utils = @import("utils.zig");

fn print(comptime fmt: []const u8, args: anytype) void {
    var buf: [1024]u8 = undefined;
    const s = std.fmt.bufPrint(&buf, fmt, args) catch return;
    const out = std.fs.File.stdout();
    out.writeAll(s) catch {};
}

fn write(s: []const u8) void {
    const out = std.fs.File.stdout();
    out.writeAll(s) catch {};
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var shelf = models.Shelf.init("Computer Science", 10);
    defer shelf.deinit(allocator);

    const books_data = [_]models.Book{
        .{ .title = "The Art of Computer Programming", .author = "Donald Knuth", .year = 1968, .isbn = "9780201896831", .available = true },
        .{ .title = "Structure and Interpretation of Computer Programs", .author = "Harold Abelson", .year = 1996, .isbn = "9780262510875", .available = true },
        .{ .title = "Introduction to Algorithms", .author = "Thomas Cormen", .year = 2009, .isbn = "9780262033848", .available = false },
        .{ .title = "Design Patterns", .author = "Erich Gamma", .year = 1994, .isbn = "9780201633610", .available = true },
        .{ .title = "The Pragmatic Programmer", .author = "David Thomas", .year = 2019, .isbn = "9780135957059", .available = true },
    };

    for (&books_data) |b| {
        try storage.addBook(allocator, &shelf, b);
    }

    var report_buf: [4096]u8 = undefined;
    const report = storage.generateReport(&report_buf, shelf);
    write(report);
    write("\n");

    write("--- Search by author \"knuth\" ---\n");
    var author_results = try storage.findByAuthor(allocator, shelf, "knuth");
    defer author_results.deinit(allocator);
    for (author_results.items) |b| {
        var fmt_buf: [512]u8 = undefined;
        const formatted = utils.formatBook(&fmt_buf, b);
        print("  {s}\n", .{formatted});
    }
    write("\n");

    write("--- Search by year range 1990-2010 ---\n");
    var year_results = try storage.findByYearRange(allocator, shelf, 1990, 2010);
    defer year_results.deinit(allocator);
    for (year_results.items) |b| {
        var fmt_buf: [512]u8 = undefined;
        const formatted = utils.formatBook(&fmt_buf, b);
        print("  {s}\n", .{formatted});
    }
    write("\n");

    write("--- Parse CSV ---\n");
    if (utils.parseCsvLine("Clean Code,Robert Martin,2008,9780132350884,true")) |parsed| {
        var fmt_buf: [512]u8 = undefined;
        const formatted = utils.formatBook(&fmt_buf, parsed);
        print("  Parsed: {s}\n", .{formatted});
    }
    write("\n");

    write("--- ISBN Validation ---\n");
    for (&books_data) |b| {
        const status: []const u8 = if (utils.validateIsbn(b.isbn)) "valid" else "invalid";
        print("  {s}: {s}\n", .{ b.isbn, status });
    }
}
