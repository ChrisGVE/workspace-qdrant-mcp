#ifndef STORAGE_HPP
#define STORAGE_HPP

#include "models.hpp"
#include <string>
#include <vector>

void add_book(Shelf &shelf, const Book &book);
Book remove_book(Shelf &shelf, const std::string &isbn);
std::vector<Book> find_by_author(const Shelf &shelf,
                                 const std::string &author);
std::vector<Book> find_by_year_range(const Shelf &shelf,
                                     int start_year, int end_year);
std::vector<Book> sort_by_title(const Shelf &shelf);
bool is_full(const Shelf &shelf);
std::string generate_report(const Shelf &shelf);

#endif
