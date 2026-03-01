#include "utils.hpp"
#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <vector>

bool validate_isbn(const std::string &isbn) {
    if (isbn.size() != 13) return false;
    for (char c : isbn) {
        if (c < '0' || c > '9') return false;
    }
    int total = 0;
    for (int i = 0; i < 13; i++) {
        int digit = isbn[i] - '0';
        total += (i % 2 == 0) ? digit : digit * 3;
    }
    return total % 10 == 0;
}

std::string format_book(const Book &book) {
    std::ostringstream ss;
    ss << "\"" << book.title << "\" by " << book.author
       << " (" << book.year << ") [ISBN: " << book.isbn << "]";
    return ss.str();
}

Book parse_csv_line(const std::string &line) {
    std::vector<std::string> fields;
    std::stringstream ss(line);
    std::string field;
    while (std::getline(ss, field, ',')) {
        fields.push_back(field);
    }
    if (fields.size() != 5) {
        throw std::runtime_error("Expected 5 fields");
    }

    int year;
    try {
        year = std::stoi(fields[2]);
    } catch (...) {
        throw std::runtime_error("Invalid year: " + fields[2]);
    }

    std::string avail = fields[4];
    /* Trim whitespace */
    avail.erase(0, avail.find_first_not_of(' '));
    avail.erase(avail.find_last_not_of(' ') + 1);

    bool available;
    if (avail == "true") {
        available = true;
    } else if (avail == "false") {
        available = false;
    } else {
        throw std::runtime_error("Invalid boolean: " + fields[4]);
    }

    return Book(fields[0], fields[1], year, fields[3], available);
}
