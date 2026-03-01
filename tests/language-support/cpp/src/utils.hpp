#ifndef UTILS_HPP
#define UTILS_HPP

#include "models.hpp"
#include <string>

bool validate_isbn(const std::string &isbn);
std::string format_book(const Book &book);
Book parse_csv_line(const std::string &line);

#endif
