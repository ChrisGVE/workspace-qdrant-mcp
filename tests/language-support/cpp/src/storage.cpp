#include "storage.hpp"
#include <algorithm>
#include <map>
#include <sstream>
#include <stdexcept>

void add_book(Shelf &shelf, const Book &book) {
    if (is_full(shelf)) {
        throw std::runtime_error("Shelf '" + shelf.name +
                                 "' is at capacity");
    }
    shelf.books.push_back(book);
}

Book remove_book(Shelf &shelf, const std::string &isbn) {
    for (auto it = shelf.books.begin(); it != shelf.books.end(); ++it) {
        if (it->isbn == isbn) {
            Book removed = *it;
            shelf.books.erase(it);
            return removed;
        }
    }
    throw std::runtime_error("No book with ISBN '" + isbn + "' found");
}

static std::string to_lower(const std::string &s) {
    std::string result = s;
    std::transform(result.begin(), result.end(), result.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return result;
}

std::vector<Book> find_by_author(const Shelf &shelf,
                                 const std::string &author) {
    std::vector<Book> results;
    std::string query = to_lower(author);
    for (const auto &book : shelf.books) {
        if (to_lower(book.author).find(query) != std::string::npos) {
            results.push_back(book);
        }
    }
    return results;
}

std::vector<Book> find_by_year_range(const Shelf &shelf,
                                     int start_year, int end_year) {
    std::vector<Book> results;
    for (const auto &book : shelf.books) {
        if (book.year >= start_year && book.year <= end_year) {
            results.push_back(book);
        }
    }
    return results;
}

std::vector<Book> sort_by_title(const Shelf &shelf) {
    std::vector<Book> sorted = shelf.books;
    std::sort(sorted.begin(), sorted.end(),
              [](const Book &a, const Book &b) {
                  return a.title < b.title;
              });
    return sorted;
}

bool is_full(const Shelf &shelf) {
    return static_cast<int>(shelf.books.size()) >= shelf.capacity;
}

std::string generate_report(const Shelf &shelf) {
    int total = static_cast<int>(shelf.books.size());
    int available_count = 0;
    for (const auto &b : shelf.books) {
        if (b.available) available_count++;
    }
    int avail_pct = total > 0 ? (available_count * 100 / total) : 0;
    int cap_pct = shelf.capacity > 0 ? (total * 100 / shelf.capacity) : 0;

    /* Count unique authors preserving insertion order */
    std::vector<std::string> author_order;
    std::map<std::string, int> author_counts;
    for (const auto &b : shelf.books) {
        if (author_counts.find(b.author) == author_counts.end()) {
            author_order.push_back(b.author);
        }
        author_counts[b.author]++;
    }

    int min_year = shelf.books[0].year;
    int max_year = shelf.books[0].year;
    for (const auto &b : shelf.books) {
        if (b.year < min_year) min_year = b.year;
        if (b.year > max_year) max_year = b.year;
    }

    std::ostringstream ss;
    ss << "=== Library Report: " << shelf.name << " ===\n";
    ss << "Total books: " << total << "\n";
    ss << "Available: " << available_count << " / " << total
       << " (" << avail_pct << "%)\n";
    ss << "Capacity: " << total << " / " << shelf.capacity
       << " (" << cap_pct << "% full)\n";
    ss << "\n";
    ss << "Authors (" << author_order.size() << " unique):\n";
    for (const auto &author : author_order) {
        ss << "  - " << author << " (" << author_counts[author]
           << " books)\n";
    }
    ss << "\n";
    ss << "Year range: " << min_year << " - " << max_year << "\n";
    ss << "\n";
    ss << "Books by availability:\n";
    for (const auto &book : shelf.books) {
        char marker = book.available ? '+' : '-';
        ss << "  [" << marker << "] " << book.title << " by "
           << book.author << " (" << book.year << ") - "
           << book.isbn << "\n";
    }
    return ss.str();
}
