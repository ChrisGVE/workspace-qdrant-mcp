const std = @import("std");

pub const Genre = enum {
    Fiction,
    NonFiction,
    Science,
    History,
    Biography,
    Technology,
    Philosophy,
};

pub const Book = struct {
    title: []const u8,
    author: []const u8,
    year: i32,
    isbn: []const u8,
    available: bool,
};

pub const Shelf = struct {
    name: []const u8,
    capacity: usize,
    books: std.ArrayList(Book),

    pub fn init(name: []const u8, capacity: usize) Shelf {
        return Shelf{
            .name = name,
            .capacity = capacity,
            .books = std.ArrayList(Book).empty,
        };
    }

    pub fn deinit(self: *Shelf, allocator: std.mem.Allocator) void {
        self.books.deinit(allocator);
    }
};
