package main

Genre :: enum {
    Fiction,
    Non_Fiction,
    Science,
    History,
    Biography,
    Technology,
    Philosophy,
}

Book :: struct {
    title:     string,
    author:    string,
    year:      int,
    isbn:      string,
    available: bool,
}

Shelf :: struct {
    name:     string,
    capacity: int,
    books:    [dynamic]Book,
}

create_book :: proc(title, author: string, year: int,
                    isbn: string, available: bool) -> Book {
    return Book{title, author, year, isbn, available}
}

create_shelf :: proc(name: string, capacity: int) -> Shelf {
    return Shelf{name, capacity, make([dynamic]Book, 0)}
}
