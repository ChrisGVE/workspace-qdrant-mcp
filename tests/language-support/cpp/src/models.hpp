#ifndef MODELS_HPP
#define MODELS_HPP

#include <string>
#include <vector>

enum class Genre {
    Fiction,
    NonFiction,
    Science,
    History,
    Biography,
    Technology,
    Philosophy
};

struct Book {
    std::string title;
    std::string author;
    int year;
    std::string isbn;
    bool available;

    Book(std::string title, std::string author, int year,
         std::string isbn, bool available);
};

struct Shelf {
    std::string name;
    int capacity;
    std::vector<Book> books;

    Shelf(std::string name, int capacity);
};

#endif
