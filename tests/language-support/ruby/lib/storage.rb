require_relative "models"

def add_book(shelf, book)
  raise "Shelf '#{shelf.name}' is at capacity" if full?(shelf)
  shelf.books << book
end

def remove_book(shelf, isbn)
  index = shelf.books.index { |b| b.isbn == isbn }
  raise "No book with ISBN '#{isbn}' found" if index.nil?
  shelf.books.delete_at(index)
end

def find_by_author(shelf, author)
  query = author.downcase
  shelf.books.select { |b| b.author.downcase.include?(query) }
end

def find_by_year_range(shelf, start_year, end_year)
  shelf.books.select { |b| b.year >= start_year && b.year <= end_year }
end

def sort_by_title(shelf)
  shelf.books.sort_by(&:title)
end

def full?(shelf)
  shelf.books.size >= shelf.capacity
end

def generate_report(shelf)
  total = shelf.books.size
  available_count = shelf.books.count(&:available)
  avail_pct = total > 0 ? (available_count * 100 / total) : 0
  cap_pct = shelf.capacity > 0 ? (total * 100 / shelf.capacity) : 0

  author_counts = {}
  shelf.books.each do |book|
    author_counts[book.author] = (author_counts[book.author] || 0) + 1
  end

  years = shelf.books.map(&:year)
  min_year = years.min
  max_year = years.max

  lines = []
  lines << "=== Library Report: #{shelf.name} ==="
  lines << "Total books: #{total}"
  lines << "Available: #{available_count} / #{total} (#{avail_pct}%)"
  lines << "Capacity: #{total} / #{shelf.capacity} (#{cap_pct}% full)"
  lines << ""
  lines << "Authors (#{author_counts.size} unique):"
  author_counts.each do |author, count|
    lines << "  - #{author} (#{count} books)"
  end
  lines << ""
  lines << "Year range: #{min_year} - #{max_year}"
  lines << ""
  lines << "Books by availability:"
  shelf.books.each do |book|
    marker = book.available ? "+" : "-"
    lines << "  [#{marker}] #{book.title} by #{book.author} (#{book.year}) - #{book.isbn}"
  end
  lines.join("\n")
end
