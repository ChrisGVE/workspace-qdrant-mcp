#include "models.hpp"
#include <utility>

Book::Book(std::string title, std::string author, int year,
           std::string isbn, bool available)
    : title(std::move(title)), author(std::move(author)), year(year),
      isbn(std::move(isbn)), available(available) {}

Shelf::Shelf(std::string name, int capacity)
    : name(std::move(name)), capacity(capacity) {}
